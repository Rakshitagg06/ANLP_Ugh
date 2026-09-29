"""Aggregate out-of-fold predictions into tables and comparison graphs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import calculate_metrics


def read_prediction_files(paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    return records


def aggregate_by_model_seed(records: List[Dict[str, Any]]) -> pd.DataFrame:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("final_det_prediction") is None or record.get("final_type_predictions") is None:
            continue
        groups[(record["model"], int(record["seed"]))].append(record)

    rows = []
    for (model, seed), group in sorted(groups.items()):
        ids = [item["example_id"] for item in group]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate example IDs for model={model}, seed={seed}")
        metrics = calculate_metrics(
            np.asarray([item["gold_det"] for item in group]),
            np.asarray([item["gold_types"] for item in group]),
            np.asarray([item["final_det_prediction"] for item in group]),
            np.asarray([item["final_type_predictions"] for item in group]),
            True,
        )
        rows.append(
            {
                "model": model,
                "seed": seed,
                "examples": len(group),
                **{key: value for key, value in metrics.items() if isinstance(value, float)},
            }
        )
    return pd.DataFrame(rows)


def save_summary_outputs(frame: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "seed_level_metrics.csv", index=False)
    numeric = [column for column in frame.columns if column not in {"model", "seed"}]
    summary = frame.groupby("model")[numeric].agg(["mean", "std"])
    summary.to_csv(output / "model_summary.csv")

    metrics = [name for name in ("det_macro_f1", "type_macro_f1", "type_macro_recall", "lvr_total") if name in frame]
    if not metrics:
        return
    figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        groups = [part[metric].dropna().to_numpy() for _, part in frame.groupby("model")]
        labels = [name for name, _ in frame.groupby("model")]
        axis.boxplot(groups, labels=labels, showmeans=True)
        axis.set_title(metric.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "model_comparison.png", dpi=180)
    plt.close(figure)

