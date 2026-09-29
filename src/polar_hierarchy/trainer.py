"""Shared training, validation, checkpointing, prediction, logging, and plotting."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import save_resolved_config
from .constants import TYPE_LABELS
from .data import PolarCollator, PolarDataset, split_outer_fold, write_jsonl
from .metrics import (
    calculate_metrics,
    hard_predictions,
    select_det_threshold,
    select_type_thresholds,
)
from .models import build_model
from .plots import plot_per_label_f1, plot_probability_histograms, plot_training_history
from .tracking import ExperimentTracker
from .utils import JsonlLogger, choose_device, environment_metadata


class Trainer:
    def __init__(
        self,
        config: Dict[str, Any],
        frame: pd.DataFrame,
        run_dir: Path,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.frame = frame
        self.run_dir = run_dir
        self.logger = logger
        self.events = JsonlLogger(run_dir / "logs" / "events.jsonl")
        self.device = choose_device(config["training"].get("device", "auto"))
        self.model_type = config["experiment"]["model_type"]
        self.fold = int(config["experiment"]["fold"])
        self.seed = int(config["experiment"]["seed"])
        self.run_name = f"{self.model_type}-fold{self.fold}-seed{self.seed}"
        self.tracker = ExperimentTracker(config, self.run_name)
        self.keep_checkpoint = bool(config["training"].get("keep_checkpoint", False))
        self._best_state: Dict[str, Any] | None = None

        self.train_frame, self.validation_frame, self.test_frame = split_outer_fold(
            frame,
            fold=self.fold,
            validation_fraction=float(config["data"].get("internal_validation_fraction", 0.1)),
            seed=self.seed,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["pretrained_name"], revision=config["model"].get("revision")
        )
        self.positive_weights = self._calculate_positive_weights()
        # Some Hugging Face checkpoints are stored or inferred as FP16. AMP
        # requires FP32 master parameters; autocast performs the FP16 forward
        # pass while GradScaler safely unscales FP32 gradients. Normalizing here
        # avoids "Attempting to unscale FP16 gradients" on Ada.
        self.model = build_model(config, self.positive_weights).to(
            device=self.device, dtype=torch.float32
        )
        parameter_dtypes = sorted(
            {
                str(parameter.dtype)
                for parameter in self.model.parameters()
                if parameter.is_floating_point()
            }
        )
        self.logger.info("Model parameter dtypes: %s", ", ".join(parameter_dtypes))
        self.loaders = self._build_loaders()
        self.det_threshold = float(config["thresholds"].get("default_det", 0.5))
        self.type_thresholds = np.repeat(
            float(config["thresholds"].get("default_type", 0.5)), len(TYPE_LABELS)
        )

    def _calculate_positive_weights(self) -> torch.Tensor | None:
        if not self.config["training"].get("use_type_positive_weights", False):
            return None
        positive = self.train_frame[self.train_frame["gold_det"] == 1]
        labels = positive[[f"type_{label}" for label in TYPE_LABELS]].to_numpy(np.float32)
        positives = labels.sum(axis=0)
        negatives = len(labels) - positives
        weights = negatives / np.clip(positives, 1.0, None)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _build_loaders(self) -> Dict[str, DataLoader]:
        training = self.config["training"]
        max_length = int(self.config["data"].get("max_length", 256))
        collator = PolarCollator(
            self.tokenizer,
            pad_to_multiple_of=8 if training.get("mixed_precision", True) else None,
        )
        loaders: Dict[str, DataLoader] = {}
        for name, frame, shuffle, batch_size in (
            ("train", self.train_frame, True, int(training["batch_size"])),
            ("validation", self.validation_frame, False, int(training["eval_batch_size"])),
            ("test", self.test_frame, False, int(training["eval_batch_size"])),
        ):
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            loaders[name] = DataLoader(
                PolarDataset(frame, self.tokenizer, max_length),
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collator,
                num_workers=int(training.get("num_workers", 0)),
                pin_memory=self.device.type == "cuda",
                generator=generator,
            )
        return loaders

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting %s on %s", self.run_name, self.device)
        self.logger.info(
            "Split sizes: train=%d validation=%d outer_test=%d",
            len(self.train_frame),
            len(self.validation_frame),
            len(self.test_frame),
        )
        save_resolved_config(self.config, self.run_dir / "resolved_config.json")
        self._save_json(environment_metadata(), self.run_dir / "environment.json")

        optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"].get("weight_decay", 0.01)),
        )
        accumulation = int(self.config["training"].get("gradient_accumulation_steps", 1))
        epochs = int(self.config["training"]["epochs"])
        steps_per_epoch = math.ceil(len(self.loaders["train"]) / accumulation)
        total_steps = max(1, steps_per_epoch * epochs)
        warmup_steps = int(total_steps * float(self.config["training"].get("warmup_ratio", 0.1)))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        amp_enabled = (
            bool(self.config["training"].get("mixed_precision", True))
            and self.device.type == "cuda"
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        selection_metric = self.config["training"].get("selection_metric") or (
            "det_macro_f1" if self.model_type == "m1_det" else "type_macro_f1"
        )
        best_score = -float("inf")
        best_epoch = 0
        patience = int(self.config["training"].get("early_stopping_patience", 2))
        stale_epochs = 0
        history: List[Dict[str, Any]] = []
        started = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(optimizer, scheduler, scaler, accumulation, amp_enabled)
            validation = self.predict(self.loaders["validation"])
            self._select_thresholds(validation)
            validation_metrics = self._score_predictions(validation)
            validation_loss = float(validation["mean_loss"])
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **{f"validation_{key}": value for key, value in validation_metrics.items() if isinstance(value, float)},
            }
            history.append(record)
            self.events.log("epoch_end", record)
            self.tracker.log(record, step=epoch)
            self.logger.info("Epoch %d | %s", epoch, json.dumps(record, sort_keys=True))

            score = float(validation_metrics.get(selection_metric, -validation_loss))
            if score > best_score:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                self._save_checkpoint(epoch, best_score)
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    self.logger.info("Early stopping after epoch %d", epoch)
                    break

        plot_training_history(history, self.run_dir / "plots")
        self._load_best_checkpoint()
        outer_predictions = self.predict(self.loaders["test"])
        outer_metrics = self._score_predictions(outer_predictions)
        runtime_seconds = time.time() - started
        summary = {
            **outer_metrics,
            "best_epoch": best_epoch,
            "best_validation_score": best_score,
            "runtime_seconds": runtime_seconds,
            "checkpoint_retained": self.keep_checkpoint,
            "det_threshold": self.det_threshold,
            "type_thresholds": self.type_thresholds.tolist(),
        }
        self._save_json(summary, self.run_dir / "metrics" / "outer_fold_metrics.json")
        self._save_prediction_records(outer_predictions)
        plot_per_label_f1(outer_metrics, self.run_dir / "plots")
        plot_probability_histograms(
            outer_predictions["det_probabilities"],
            outer_predictions["type_probabilities"],
            self.run_dir / "plots",
        )
        self.events.log("run_complete", summary)
        tracked_summary = {
            f"outer/{key}": value
            for key, value in summary.items()
            if isinstance(value, (int, float))
        }
        for label, label_metrics in summary.get("per_label", {}).items():
            for metric, value in label_metrics.items():
                if isinstance(value, (int, float)):
                    tracked_summary[f"outer/per_label/{label}/{metric}"] = value
        self.tracker.log(tracked_summary)
        self.tracker.log_images(
            {
                "plots/loss_curves": self.run_dir / "plots" / "loss_curves.png",
                "plots/metric_curves": self.run_dir / "plots" / "metric_curves.png",
                "plots/per_label_f1": self.run_dir / "plots" / "per_label_f1.png",
                "plots/det_probability_histogram": self.run_dir
                / "plots"
                / "det_probability_histogram.png",
                "plots/type_probability_histograms": self.run_dir
                / "plots"
                / "type_probability_histograms.png",
            }
        )
        self.tracker.finish()
        self.logger.info("Completed %s | %s", self.run_name, json.dumps(summary, sort_keys=True))
        return summary

    def _train_epoch(
        self,
        optimizer: AdamW,
        scheduler: Any,
        scaler: torch.amp.GradScaler,
        accumulation: int,
        amp_enabled: bool,
    ) -> float:
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches = 0
        for batch_index, batch in enumerate(self.loaders["train"], start=1):
            model_inputs = self._to_device(batch)
            with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                output = self.model(**model_inputs)
                if output.loss is None:
                    raise RuntimeError("Training model returned no loss")
                loss = output.loss / accumulation
            scaler.scale(loss).backward()
            total_loss += float(output.loss.detach().cpu())
            batches += 1
            if batch_index % accumulation == 0 or batch_index == len(self.loaders["train"]):
                scaler.unscale_(optimizer)
                clip_grad_norm_(
                    self.model.parameters(),
                    float(self.config["training"].get("max_grad_norm", 1.0)),
                )
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before_step:
                    scheduler.step()
                else:
                    self.events.log(
                        "amp_step_skipped",
                        {"batch": batch_index, "scale": scaler.get_scale()},
                    )
                optimizer.zero_grad(set_to_none=True)
        return total_loss / max(batches, 1)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        ids: List[str] = []
        gold_det: List[np.ndarray] = []
        gold_types: List[np.ndarray] = []
        det_probabilities: List[np.ndarray] = []
        type_probabilities: List[np.ndarray] = []
        losses: List[float] = []
        for batch in loader:
            ids.extend(batch["example_ids"])
            gold_det.append(batch["det_labels"].numpy())
            gold_types.append(batch["type_labels"].numpy())
            output = self.model(**self._to_device(batch))
            if output.loss is not None:
                losses.append(float(output.loss.detach().cpu()))
            if output.det_logits is not None:
                det_probabilities.append(torch.sigmoid(output.det_logits).cpu().numpy())
            if output.type_logits is not None:
                type_probabilities.append(torch.sigmoid(output.type_logits).cpu().numpy())
        return {
            "example_ids": ids,
            "gold_det": np.concatenate(gold_det),
            "gold_types": np.concatenate(gold_types),
            "det_probabilities": np.concatenate(det_probabilities) if det_probabilities else None,
            "type_probabilities": np.concatenate(type_probabilities) if type_probabilities else None,
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
        }

    def _select_thresholds(self, predictions: Dict[str, Any]) -> None:
        thresholds = self.config["thresholds"]
        if predictions["det_probabilities"] is not None and self.model_type != "m4_core":
            self.det_threshold = select_det_threshold(
                predictions["gold_det"], predictions["det_probabilities"], thresholds["det_grid"]
            )
        if predictions["type_probabilities"] is not None:
            self.type_thresholds = select_type_thresholds(
                predictions["gold_det"],
                predictions["gold_types"],
                predictions["type_probabilities"],
                thresholds["type_grid"],
                thresholds.get("type_mode", "per_label"),
            )

    def _score_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        pred_det, pred_types = hard_predictions(
            predictions["det_probabilities"],
            predictions["type_probabilities"],
            self.det_threshold,
            self.type_thresholds,
            self.model_type,
        )
        return calculate_metrics(
            predictions["gold_det"],
            predictions["gold_types"],
            pred_det,
            pred_types,
            bool(self.config["data"].get("type_metrics_on_gold_det_positive_only", True)),
        )

    def _save_checkpoint(self, epoch: int, score: float) -> None:
        state = {
            "model_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.model.state_dict().items()
            },
            "epoch": epoch,
            "score": score,
            "det_threshold": self.det_threshold,
            "type_thresholds": self.type_thresholds.tolist(),
        }
        if self.keep_checkpoint:
            checkpoint_dir = self.run_dir / "checkpoints" / "best"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(state, checkpoint_dir / "training_state.pt")
            self.tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
        else:
            self._best_state = state

    def _load_best_checkpoint(self) -> None:
        if self._best_state is not None:
            state = self._best_state
        else:
            checkpoint_path = self.run_dir / "checkpoints" / "best" / "training_state.pt"
            state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.det_threshold = float(state["det_threshold"])
        self.type_thresholds = np.asarray(state["type_thresholds"], dtype=np.float64)
        self._best_state = None

    def _save_prediction_records(self, predictions: Dict[str, Any]) -> None:
        pred_det, pred_types = hard_predictions(
            predictions["det_probabilities"],
            predictions["type_probabilities"],
            self.det_threshold,
            self.type_thresholds,
            self.model_type,
        )
        records = []
        for index, example_id in enumerate(predictions["example_ids"]):
            raw_det_prediction = None
            if predictions["det_probabilities"] is not None and self.model_type != "m4_core":
                raw_det_prediction = int(
                    predictions["det_probabilities"][index] >= self.det_threshold
                )
            raw_type_predictions = None
            if predictions["type_probabilities"] is not None:
                raw_type_predictions = (
                    predictions["type_probabilities"][index] >= self.type_thresholds
                ).astype(int).tolist()
            records.append(
                {
                    "example_id": example_id,
                    "fold": self.fold,
                    "seed": self.seed,
                    "model": self.model_type,
                    "gold_det": int(predictions["gold_det"][index]),
                    "gold_types": predictions["gold_types"][index].astype(int).tolist(),
                    "raw_det_probability": None
                    if predictions["det_probabilities"] is None
                    else float(predictions["det_probabilities"][index]),
                    "raw_det_prediction": raw_det_prediction,
                    "type_probabilities": None
                    if predictions["type_probabilities"] is None
                    else predictions["type_probabilities"][index].tolist(),
                    "raw_type_predictions": raw_type_predictions,
                    "final_det_prediction": None if pred_det is None else int(pred_det[index]),
                    "final_type_predictions": None
                    if pred_types is None
                    else pred_types[index].astype(int).tolist(),
                    "det_threshold": self.det_threshold if pred_det is not None else None,
                    "type_thresholds": self.type_thresholds.tolist() if pred_types is not None else None,
                }
            )
        write_jsonl(records, self.run_dir / "predictions" / "outer_fold_predictions.jsonl")

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
            if key != "example_ids"
        }

    @staticmethod
    def _save_json(value: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
