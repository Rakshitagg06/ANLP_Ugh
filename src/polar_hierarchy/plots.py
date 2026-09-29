"""Non-interactive experiment plots suitable for Ada jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import TYPE_LABELS


def plot_training_history(history: List[Dict[str, Any]], output_dir: str | Path) -> None:
    if not history:
        return
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    epochs = [record["epoch"] for record in history]

    figure, axis = plt.subplots(figsize=(7, 4))
    for key, label in (("train_loss", "Train"), ("validation_loss", "Validation")):
        values = [record.get(key, np.nan) for record in history]
        axis.plot(epochs, values, marker="o", label=label)
    axis.set(xlabel="Epoch", ylabel="Loss", title="Training history")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "loss_curves.png", dpi=180)
    plt.close(figure)

    metric_keys = sorted(
        {key for record in history for key in record if key.startswith("validation_") and key != "validation_loss"}
    )
    if metric_keys:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for key in metric_keys:
            values = [record.get(key, np.nan) for record in history]
            axis.plot(epochs, values, marker="o", label=key.removeprefix("validation_"))
        axis.set(xlabel="Epoch", ylabel="Score", title="Validation metrics")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
        figure.tight_layout()
        figure.savefig(output / "metric_curves.png", dpi=180)
        plt.close(figure)


def plot_per_label_f1(metrics: Dict[str, Any], output_dir: str | Path) -> None:
    if "per_label" not in metrics:
        return
    values = [metrics["per_label"][label]["f1"] for label in TYPE_LABELS]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(TYPE_LABELS, values, color="#355C7D")
    axis.set_ylim(0, 1)
    axis.set_ylabel("F1")
    axis.set_title("Per-label TYPE F1")
    axis.tick_params(axis="x", rotation=25)
    axis.bar_label(bars, fmt="%.3f", padding=2)
    figure.tight_layout()
    figure.savefig(Path(output_dir) / "per_label_f1.png", dpi=180)
    plt.close(figure)


def plot_probability_histograms(
    det_probabilities: np.ndarray | None,
    type_probabilities: np.ndarray | None,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    if det_probabilities is not None:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.hist(det_probabilities, bins=20, color="#6C5B7B", alpha=0.85)
        axis.set(xlabel="DET probability", ylabel="Count", title="DET probability distribution")
        figure.tight_layout()
        figure.savefig(output / "det_probability_histogram.png", dpi=180)
        plt.close(figure)
    if type_probabilities is not None:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for index, label in enumerate(TYPE_LABELS):
            axis.hist(type_probabilities[:, index], bins=20, alpha=0.45, label=label)
        axis.set(xlabel="TYPE probability", ylabel="Count", title="TYPE probability distributions")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output / "type_probability_histograms.png", dpi=180)
        plt.close(figure)

