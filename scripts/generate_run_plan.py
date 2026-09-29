#!/usr/bin/env python3
"""Generate immutable Ada run-plan TSVs for training and CPU post-processing."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SEEDS = (13, 21, 42, 87, 100)
MODELS = (
    ("m1_det", "configs/m1_det.yaml"),
    ("m1_type", "configs/m1_type.yaml"),
    ("m2", "configs/m2.yaml"),
    ("m4_core", "configs/m4_core.yaml"),
)


def write_training_plan(path: Path) -> None:
    rows = []
    task_id = 0
    # Interleave models so every eight-task wave contains all model families.
    for fold in range(5):
        for seed in SEEDS:
            for model, config in MODELS:
                rows.append((task_id, model, config, fold, seed))
                task_id += 1
    write_tsv(path, ("task_id", "model", "config", "fold", "seed"), rows)


def write_m3_plan(path: Path) -> None:
    rows = []
    task_id = 0
    for fold in range(5):
        for seed in SEEDS:
            rows.append((task_id, fold, seed))
            task_id += 1
    write_tsv(path, ("task_id", "fold", "seed"), rows)


def write_m1_plan(path: Path) -> None:
    rows = []
    task_id = 0
    for fold in range(5):
        for seed in SEEDS:
            rows.append((task_id, fold, seed))
            task_id += 1
    write_tsv(path, ("task_id", "fold", "seed"), rows)


def write_tsv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="slurm/run_plans")
    args = parser.parse_args()
    output = Path(args.output_dir)
    write_training_plan(output / "training.tsv")
    write_m3_plan(output / "m3.tsv")
    write_m1_plan(output / "m1.tsv")
    print(f"Wrote 100 training tasks to {output / 'training.tsv'}")
    print(f"Wrote 25 M3 tasks to {output / 'm3.tsv'}")
    print(f"Wrote 25 M1 combination tasks to {output / 'm1.tsv'}")


if __name__ == "__main__":
    main()
