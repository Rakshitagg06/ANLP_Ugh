"""Threshold selection and evaluation metrics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .constants import TYPE_LABELS


def hard_predictions(
    det_probabilities: np.ndarray | None,
    type_probabilities: np.ndarray | None,
    det_threshold: float,
    type_thresholds: Iterable[float],
    model_type: str,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    type_predictions = None
    if type_probabilities is not None:
        thresholds = np.asarray(list(type_thresholds), dtype=np.float64).reshape(1, -1)
        type_predictions = (type_probabilities >= thresholds).astype(np.int64)

    if model_type == "m4_core":
        if type_predictions is None:
            raise ValueError("M4-core requires TYPE probabilities")
        det_predictions = type_predictions.any(axis=1).astype(np.int64)
    elif det_probabilities is not None:
        det_predictions = (det_probabilities >= det_threshold).astype(np.int64)
    else:
        det_predictions = None
    return det_predictions, type_predictions


def calculate_metrics(
    gold_det: np.ndarray,
    gold_types: np.ndarray,
    pred_det: np.ndarray | None,
    pred_types: np.ndarray | None,
    type_on_gold_positive_only: bool = True,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if pred_det is not None:
        metrics["det_macro_f1"] = float(f1_score(gold_det, pred_det, average="macro", zero_division=0))

    if pred_types is not None:
        type_mask = gold_det.astype(bool) if type_on_gold_positive_only else np.ones(len(gold_det), bool)
        if not np.any(type_mask):
            raise ValueError("No examples are available for TYPE evaluation")
        y_true = gold_types[type_mask]
        y_pred = pred_types[type_mask]
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        metrics.update(
            {
                "type_macro_precision": float(np.mean(precision)),
                "type_macro_recall": float(np.mean(recall)),
                "type_macro_f1": float(np.mean(f1)),
            }
        )
        metrics["per_label"] = {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(TYPE_LABELS)
        }

    if pred_det is not None and pred_types is not None:
        metrics.update(logical_violation_metrics(pred_det, pred_types))
    return metrics


def logical_violation_metrics(pred_det: np.ndarray, pred_types: np.ndarray) -> Dict[str, float]:
    any_type = pred_types.astype(bool).any(axis=1)
    det = pred_det.astype(bool)
    violation_a = (~det) & any_type
    violation_b = det & (~any_type)
    return {
        "lvr_a": float(np.mean(violation_a)),
        "lvr_b": float(np.mean(violation_b)),
        "lvr_total": float(np.mean(violation_a | violation_b)),
    }


def select_det_threshold(
    gold_det: np.ndarray,
    probabilities: np.ndarray,
    grid: Iterable[float],
) -> float:
    candidates = list(grid)
    if not candidates:
        raise ValueError("DET threshold grid is empty")
    scored = [
        (float(f1_score(gold_det, probabilities >= value, average="macro", zero_division=0)), value)
        for value in candidates
    ]
    return float(max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1])


def select_type_thresholds(
    gold_det: np.ndarray,
    gold_types: np.ndarray,
    probabilities: np.ndarray,
    grid: Iterable[float],
    mode: str,
) -> np.ndarray:
    candidates = list(grid)
    if not candidates:
        raise ValueError("TYPE threshold grid is empty")
    mask = gold_det.astype(bool)
    y_true = gold_types[mask]
    y_prob = probabilities[mask]
    if mode == "shared":
        scored = [
            (
                float(f1_score(y_true, y_prob >= value, average="macro", zero_division=0)),
                value,
            )
            for value in candidates
        ]
        best = max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1]
        return np.repeat(float(best), y_true.shape[1])
    if mode != "per_label":
        raise ValueError(f"Unknown TYPE threshold mode: {mode}")

    thresholds = []
    for index in range(y_true.shape[1]):
        scored = [
            (
                float(f1_score(y_true[:, index], y_prob[:, index] >= value, zero_division=0)),
                value,
            )
            for value in candidates
        ]
        thresholds.append(max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1])
    return np.asarray(thresholds, dtype=np.float64)

