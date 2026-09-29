"""M3 post-hoc hierarchy reconciliation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def reconcile_one_way(det_predictions: np.ndarray, type_predictions: np.ndarray) -> np.ndarray:
    """Remove TYPE labels whenever raw DET is negative."""
    return type_predictions.astype(np.int64) * det_predictions.astype(np.int64)[:, None]


def reconcile_symmetric(
    det_predictions: np.ndarray, type_predictions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Gate TYPE using DET, then derive final DET as OR(TYPE)."""
    final_types = reconcile_one_way(det_predictions, type_predictions)
    final_det = final_types.any(axis=1).astype(np.int64)
    return final_det, final_types


def gating_audit(
    gold_types: np.ndarray,
    raw_det: np.ndarray,
    raw_types: np.ndarray,
    final_det: np.ndarray,
    final_types: np.ndarray,
) -> Dict[str, int]:
    removed = (raw_types == 1) & (final_types == 0)
    return {
        "predicted_type_labels_removed": int(removed.sum()),
        "true_positive_type_labels_removed": int((removed & (gold_types == 1)).sum()),
        "false_positive_type_labels_removed": int((removed & (gold_types == 0)).sum()),
        "det_predictions_changed": int((raw_det != final_det).sum()),
    }

