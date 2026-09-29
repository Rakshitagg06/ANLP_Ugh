"""Tests for quota-safe best-state handling."""

from __future__ import annotations

import numpy as np
import torch

from polar_hierarchy.trainer import Trainer


def test_ephemeral_best_state_restores_without_writing_checkpoint(tmp_path) -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.model = torch.nn.Linear(3, 2)
    trainer.keep_checkpoint = False
    trainer._best_state = None
    trainer.run_dir = tmp_path
    trainer.det_threshold = 0.4
    trainer.type_thresholds = np.asarray([0.2, 0.3, 0.4, 0.5, 0.6])

    expected = {
        name: tensor.detach().clone() for name, tensor in trainer.model.state_dict().items()
    }
    trainer._save_checkpoint(epoch=2, score=0.75)

    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.zero_()
    trainer.det_threshold = 0.9
    trainer.type_thresholds[:] = 0.9

    trainer._load_best_checkpoint()

    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(tensor, expected[name])
    assert trainer.det_threshold == 0.4
    assert np.allclose(trainer.type_thresholds, [0.2, 0.3, 0.4, 0.5, 0.6])
    assert trainer._best_state is None
    assert not (tmp_path / "checkpoints" / "best" / "training_state.pt").exists()
