#!/usr/bin/env python3
"""Aggregate complete out-of-fold prediction files and generate comparison plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polar_hierarchy.analysis import aggregate_by_model_seed, read_prediction_files, save_summary_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", help="JSONL files containing complete predictions")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()
    frame = aggregate_by_model_seed(read_prediction_files(args.predictions))
    if frame.empty:
        raise ValueError("No complete DET+TYPE prediction records were found")
    save_summary_outputs(frame, args.output_dir)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
