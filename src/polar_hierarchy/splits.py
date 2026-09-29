"""Reproducible multilabel fold creation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .constants import TYPE_LABELS
from .data import canonicalize_frame, read_table


def create_folds(data_config: Dict[str, Any], output_path: str | Path, seed: int) -> pd.DataFrame:
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    except ImportError as exc:
        raise RuntimeError(
            "Install iterative-stratification before generating folds: pip install -e '.[dev]'"
        ) from exc

    raw = read_table(data_config["path"])
    config_without_fold = dict(data_config)
    config_without_fold["fold_column"] = None
    canonical = canonicalize_frame(raw, config_without_fold)
    label_columns = ["gold_det", *[f"type_{name}" for name in TYPE_LABELS]]
    targets = canonical[label_columns].to_numpy(dtype=np.int64)
    splitter = MultilabelStratifiedKFold(
        n_splits=int(data_config.get("num_folds", 5)),
        shuffle=True,
        random_state=seed,
    )
    fold_ids = np.full(len(canonical), -1, dtype=np.int64)
    for fold, (_, test_indices) in enumerate(splitter.split(np.zeros(len(canonical)), targets)):
        fold_ids[test_indices] = fold
    canonical["fold"] = fold_ids
    if (canonical["fold"] < 0).any():
        raise RuntimeError("Some examples did not receive a fold")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(destination, index=False)
    return canonical
