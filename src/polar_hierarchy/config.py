"""Configuration loading, validation, and command-line overrides."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

from .constants import MODEL_TYPES, TYPE_LABELS


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    """Load YAML, apply dotted ``key=value`` overrides, and validate it."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    config = copy.deepcopy(loaded)
    for expression in overrides:
        _apply_override(config, expression)
    validate_config(config)
    config["_config_path"] = str(config_path.resolve())
    return config


def _apply_override(config: Dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ConfigError(f"Override must have key=value form: {expression!r}")
    dotted_key, raw_value = expression.split("=", 1)
    keys = dotted_key.split(".")
    target = config
    for key in keys[:-1]:
        current = target.get(key)
        if current is None:
            current = {}
            target[key] = current
        if not isinstance(current, dict):
            raise ConfigError(f"Cannot descend into non-mapping key: {dotted_key!r}")
        target = current
    target[keys[-1]] = yaml.safe_load(raw_value)


def validate_config(config: Dict[str, Any]) -> None:
    required_sections = ("experiment", "data", "model", "training", "thresholds", "logging")
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ConfigError(f"Missing configuration sections: {missing}")

    model_type = config["experiment"].get("model_type")
    if model_type not in MODEL_TYPES:
        raise ConfigError(f"experiment.model_type must be one of {MODEL_TYPES}, got {model_type!r}")

    type_columns = config["data"].get("type_label_columns", {})
    missing_labels = [label for label in TYPE_LABELS if label not in type_columns]
    if missing_labels:
        raise ConfigError(f"Missing TYPE column mappings: {missing_labels}")

    folds = int(config["data"].get("num_folds", 5))
    fold = int(config["experiment"].get("fold", 0))
    if fold < 0 or fold >= folds:
        raise ConfigError(f"fold must be in [0, {folds - 1}], got {fold}")

    if int(config["training"].get("gradient_accumulation_steps", 1)) < 1:
        raise ConfigError("gradient_accumulation_steps must be at least one")


def save_resolved_config(config: Dict[str, Any], path: str | Path) -> None:
    """Persist a JSON-safe resolved configuration."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write("\n")

