#!/usr/bin/env python3
"""Create deterministic multilabel-stratified outer folds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polar_hierarchy.config import load_config
from polar_hierarchy.splits import create_folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    config = load_config(args.config)
    frame = create_folds(config["data"], args.output, args.seed)
    label_columns = [column for column in frame.columns if column.startswith("type_")]
    print(frame.groupby("fold")[["gold_det", *label_columns]].sum().to_string())


if __name__ == "__main__":
    main()

