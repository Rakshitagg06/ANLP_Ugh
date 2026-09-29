#!/usr/bin/env python3
"""Create M3 one-way and symmetric predictions from one M2 prediction file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polar_hierarchy.data import write_jsonl
from polar_hierarchy.metrics import calculate_metrics
from polar_hierarchy.reconciliation import gating_audit, reconcile_one_way, reconcile_symmetric


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    records = read_jsonl(Path(args.m2_predictions))
    if not records:
        raise ValueError("M2 prediction file is empty")
    raw_det = np.asarray([item["final_det_prediction"] for item in records], dtype=np.int64)
    raw_types = np.asarray([item["final_type_predictions"] for item in records], dtype=np.int64)
    gold_det = np.asarray([item["gold_det"] for item in records], dtype=np.int64)
    gold_types = np.asarray([item["gold_types"] for item in records], dtype=np.int64)

    one_way_types = reconcile_one_way(raw_det, raw_types)
    symmetric_det, symmetric_types = reconcile_symmetric(raw_det, raw_types)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = {
        "m3_one_way": (raw_det, one_way_types),
        "m3_symmetric": (symmetric_det, symmetric_types),
    }
    all_metrics = {}
    for condition, (det_predictions, type_predictions) in conditions.items():
        output_records = []
        for index, source in enumerate(records):
            output_records.append(
                {
                    **source,
                    "model": condition,
                    "raw_det_prediction": int(raw_det[index]),
                    "raw_type_predictions": raw_types[index].tolist(),
                    "final_det_prediction": int(det_predictions[index]),
                    "final_type_predictions": type_predictions[index].tolist(),
                }
            )
        write_jsonl(output_records, output_dir / f"{condition}_predictions.jsonl")
        all_metrics[condition] = calculate_metrics(
            gold_det, gold_types, det_predictions, type_predictions, True
        )

    all_metrics["gating_audit"] = gating_audit(
        gold_types, raw_det, raw_types, symmetric_det, symmetric_types
    )
    with (output_dir / "m3_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(all_metrics, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(all_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

