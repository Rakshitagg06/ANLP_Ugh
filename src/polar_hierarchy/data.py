"""Canonical data loading, validation, tokenization, and fold splitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import TYPE_LABELS


class DataValidationError(ValueError):
    """Raised when input data violates the documented schema."""


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(source, lines=True)
    if suffix == ".json":
        return pd.read_json(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    raise DataValidationError(f"Unsupported data format {suffix!r}: {source}")


def canonicalize_frame(frame: pd.DataFrame, data_config: Dict[str, Any]) -> pd.DataFrame:
    """Map task-specific source columns into the internal canonical schema."""
    type_mapping = data_config["type_label_columns"]
    rename_map = {
        data_config["id_column"]: "example_id",
        data_config["text_column"]: "text",
        data_config["det_label_column"]: "gold_det",
        **{source: f"type_{label}" for label, source in type_mapping.items()},
    }
    fold_source = data_config.get("fold_column")
    if fold_source:
        rename_map[fold_source] = "fold"

    missing_source = [column for column in rename_map if column not in frame.columns]
    if missing_source:
        raise DataValidationError(f"Input data is missing configured columns: {missing_source}")

    canonical = frame.rename(columns=rename_map).copy()
    required = ["example_id", "text", "gold_det", *[f"type_{x}" for x in TYPE_LABELS]]
    if fold_source:
        required.append("fold")
    canonical = canonical[required]
    validate_canonical_frame(canonical, int(data_config.get("num_folds", 5)))
    return canonical


def validate_canonical_frame(frame: pd.DataFrame, num_folds: int = 5) -> None:
    if frame.empty:
        raise DataValidationError("Dataset is empty")
    if frame["example_id"].isna().any() or frame["example_id"].duplicated().any():
        raise DataValidationError("example_id must be non-null and unique")
    if frame["text"].isna().any() or (frame["text"].astype(str).str.strip() == "").any():
        raise DataValidationError("text must be non-empty")

    label_columns = ["gold_det", *[f"type_{x}" for x in TYPE_LABELS]]
    for column in label_columns:
        values = set(frame[column].dropna().astype(int).unique())
        if not values.issubset({0, 1}):
            raise DataValidationError(f"{column} must be binary; observed {sorted(values)}")

    if "fold" in frame:
        if frame["fold"].isna().any():
            raise DataValidationError("fold contains missing values")
        folds = set(frame["fold"].astype(int).unique())
        expected = set(range(num_folds))
        if folds != expected:
            raise DataValidationError(f"fold values must equal {sorted(expected)}, got {sorted(folds)}")


def load_canonical_data(data_config: Dict[str, Any]) -> pd.DataFrame:
    return canonicalize_frame(read_table(data_config["path"]), data_config)


def split_outer_fold(
    frame: pd.DataFrame,
    fold: int,
    validation_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, internal-validation, and untouched outer-test frames."""
    outer_test = frame[frame["fold"] == fold].copy()
    outer_train = frame[frame["fold"] != fold].copy()
    if outer_test.empty or outer_train.empty:
        raise DataValidationError(f"Fold {fold} produced an empty train or test partition")

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        label_columns = ["gold_det", *[f"type_{name}" for name in TYPE_LABELS]]
        targets = outer_train[label_columns].to_numpy(dtype=np.int64)
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=validation_fraction, random_state=seed
        )
        train_indices, validation_indices = next(
            splitter.split(np.zeros(len(outer_train)), targets)
        )
        train = outer_train.iloc[train_indices]
        validation = outer_train.iloc[validation_indices]
    except ImportError:
        from sklearn.model_selection import train_test_split

        stratify_key = _safe_stratification_key(outer_train)
        train, validation = train_test_split(
            outer_train,
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=stratify_key,
        )
    return train.reset_index(drop=True), validation.reset_index(drop=True), outer_test.reset_index(drop=True)


def _safe_stratification_key(frame: pd.DataFrame) -> pd.Series | None:
    """Use DET for a robust inner split; fall back to unstratified for tiny classes."""
    key = frame["gold_det"].astype(str)
    counts = key.value_counts()
    return key if len(counts) > 1 and int(counts.min()) >= 2 else None


class PolarDataset(Dataset):
    """Torch dataset retaining IDs and all labels for prediction auditing."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.ids = frame["example_id"].astype(str).tolist()
        self.texts = frame["text"].astype(str).tolist()
        self.det_labels = frame["gold_det"].astype(np.float32).to_numpy()
        self.type_labels = frame[[f"type_{x}" for x in TYPE_LABELS]].astype(np.float32).to_numpy()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["example_id"] = self.ids[index]
        encoded["det_label"] = float(self.det_labels[index])
        encoded["type_labels"] = self.type_labels[index].tolist()
        return encoded


class PolarCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        ids = [item.pop("example_id") for item in examples]
        det_labels = torch.tensor([item.pop("det_label") for item in examples], dtype=torch.float32)
        type_labels = torch.tensor([item.pop("type_labels") for item in examples], dtype=torch.float32)
        batch = self.tokenizer.pad(
            list(examples),
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["example_ids"] = ids
        batch["det_labels"] = det_labels
        batch["type_labels"] = type_labels
        return batch


def write_jsonl(records: List[Dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
