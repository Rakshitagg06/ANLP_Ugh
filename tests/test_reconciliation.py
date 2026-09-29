import numpy as np

from polar_hierarchy.metrics import logical_violation_metrics
from polar_hierarchy.reconciliation import reconcile_one_way, reconcile_symmetric


def test_one_way_can_retain_reverse_violation():
    det = np.array([0, 1])
    types = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    final_types = reconcile_one_way(det, types)
    metrics = logical_violation_metrics(det, final_types)
    assert metrics["lvr_a"] == 0.0
    assert metrics["lvr_b"] == 0.5


def test_symmetric_reconciliation_always_has_zero_lvr():
    det = np.array([0, 1, 1, 0])
    types = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    final_det, final_types = reconcile_symmetric(det, types)
    metrics = logical_violation_metrics(final_det, final_types)
    assert metrics["lvr_total"] == 0.0

