#!/usr/bin/env python3
"""Validate the canonical dataset and save a reproducible data-audit report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polar_hierarchy.config import load_config
from polar_hierarchy.constants import TYPE_LABELS
from polar_hierarchy.data import load_canonical_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/data_audit")
    args = parser.parse_args()
    config = load_config(args.config)
    frame = load_canonical_data(config["data"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    type_columns = [f"type_{label}" for label in TYPE_LABELS]
    any_type = frame[type_columns].astype(bool).any(axis=1)
    report = {
        "examples": int(len(frame)),
        "unique_ids": int(frame["example_id"].nunique()),
        "exact_duplicate_text_rows": int(frame["text"].duplicated(keep=False).sum()),
        "det_positive": int(frame["gold_det"].sum()),
        "det_negative": int((frame["gold_det"] == 0).sum()),
        "gold_det_negative_with_type": int(((frame["gold_det"] == 0) & any_type).sum()),
        "gold_det_positive_without_type": int(((frame["gold_det"] == 1) & ~any_type).sum()),
        "type_positive_counts": {
            label: int(frame[f"type_{label}"].sum()) for label in TYPE_LABELS
        },
        "other_with_named_type": int(
            (
                (frame["type_other"] == 1)
                & frame[[f"type_{label}" for label in TYPE_LABELS[:-1]]].astype(bool).any(axis=1)
            ).sum()
        ),
        "fold_sizes": {
            str(int(fold)): int(count) for fold, count in frame["fold"].value_counts().sort_index().items()
        },
        "fold_positive_counts": {
            str(int(fold)): {
                "gold_det": int(part["gold_det"].sum()),
                **{label: int(part[f"type_{label}"].sum()) for label in TYPE_LABELS},
            }
            for fold, part in frame.groupby("fold")
        },
    }
    with (output / "data_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")

    labels = ["DET", *TYPE_LABELS]
    counts = [report["det_positive"], *[report["type_positive_counts"][label] for label in TYPE_LABELS]]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    bars = axis.bar(labels, counts, color="#355C7D")
    axis.set(title="Positive label counts", ylabel="Examples")
    axis.tick_params(axis="x", rotation=25)
    axis.bar_label(bars, padding=2)
    figure.tight_layout()
    figure.savefig(output / "label_distribution.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

