#!/usr/bin/env python3
"""Train and evaluate one model/fold/seed condition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polar_hierarchy.config import load_config
from polar_hierarchy.data import load_canonical_data
from polar_hierarchy.trainer import Trainer
from polar_hierarchy.utils import configure_logging, create_run_directory, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    parser.add_argument("--fold", type=int, help="Override experiment.fold")
    parser.add_argument("--seed", type=int, help="Override experiment.seed")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional dotted configuration override; may be repeated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    if args.fold is not None:
        overrides.append(f"experiment.fold={args.fold}")
    if args.seed is not None:
        overrides.append(f"experiment.seed={args.seed}")
    config = load_config(args.config, overrides)
    seed_everything(
        int(config["experiment"]["seed"]),
        deterministic=bool(config["training"].get("deterministic", True)),
    )
    run_dir = create_run_directory(config)
    logger = configure_logging(run_dir, config["logging"].get("level", "INFO"))
    frame = load_canonical_data(config["data"])
    Trainer(config, frame, run_dir, logger).run()


if __name__ == "__main__":
    main()

