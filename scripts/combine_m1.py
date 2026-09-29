#!/usr/bin/env python3
"""Combine matching M1-DET and M1-TYPE predictions into one M1 condition."""

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


def load_index(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    return {record["example_id"]: record for record in records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det-predictions", required=True)
    parser.add_argument("--type-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metrics-output",
        help="Metrics JSON path; defaults beside --output with a .metrics.json suffix",
    )
    args = parser.parse_args()
    det = load_index(Path(args.det_predictions))
    types = load_index(Path(args.type_predictions))
    if set(det) != set(types):
        raise ValueError("M1-DET and M1-TYPE prediction IDs do not match")

    records = []
    for example_id in sorted(det):
        det_record, type_record = det[example_id], types[example_id]
        for key in ("gold_det", "gold_types", "fold", "seed"):
            if det_record[key] != type_record[key]:
                raise ValueError(f"Mismatched {key} for {example_id}")
        records.append(
            {
                **type_record,
                "model": "m1",
                "raw_det_probability": det_record["raw_det_probability"],
                "final_det_prediction": det_record["final_det_prediction"],
                "det_threshold": det_record["det_threshold"],
            }
        )
    write_jsonl(records, args.output)
    metrics = calculate_metrics(
        np.asarray([item["gold_det"] for item in records]),
        np.asarray([item["gold_types"] for item in records]),
        np.asarray([item["final_det_prediction"] for item in records]),
        np.asarray([item["final_type_predictions"] for item in records]),
        True,
    )
    metrics_path = (
        Path(args.metrics_output)
        if args.metrics_output
        else Path(args.output).with_suffix(".metrics.json")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
