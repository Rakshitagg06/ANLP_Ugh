import numpy as np

from polar_hierarchy.metrics import hard_predictions, select_type_thresholds


def test_m4_hard_detection_is_or_of_types():
    type_probabilities = np.array([[0.4, 0.4, 0.4, 0.4, 0.4], [0.7, 0.1, 0.1, 0.1, 0.1]])
    det_probabilities = np.array([0.92, 0.75])
    det, types = hard_predictions(
        det_probabilities,
        type_probabilities,
        det_threshold=0.5,
        type_thresholds=[0.5] * 5,
        model_type="m4_core",
    )
    assert det.tolist() == [0, 1]
    assert np.array_equal(det, types.any(axis=1).astype(int))


def test_per_label_threshold_selection_returns_five_values():
    gold_det = np.array([1, 1, 1, 0])
    gold_types = np.array(
        [[1, 0, 0, 0, 1], [0, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]]
    )
    probabilities = np.array(
        [[0.9, 0.2, 0.1, 0.1, 0.8], [0.2, 0.8, 0.2, 0.1, 0.1], [0.8, 0.7, 0.6, 0.1, 0.2], [0.9, 0.9, 0.9, 0.9, 0.9]]
    )
    thresholds = select_type_thresholds(
        gold_det, gold_types, probabilities, [0.3, 0.5, 0.7], "per_label"
    )
    assert thresholds.shape == (5,)

