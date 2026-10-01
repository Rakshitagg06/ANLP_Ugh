import numpy as np
import torch

from polar_hierarchy.losses import noisy_or_logit, noisy_or_probability


def test_noisy_or_matches_direct_probability():
    logits = torch.tensor([[0.0, -1.0, 2.0, -0.5, 0.7]], dtype=torch.float64)
    probabilities = torch.sigmoid(logits)
    direct = 1.0 - torch.prod(1.0 - probabilities, dim=-1)
    actual = noisy_or_probability(logits)
    assert torch.allclose(actual, direct, atol=1e-10)


def test_noisy_or_logit_maps_to_probability():
    logits = torch.tensor([[-10.0] * 5, [10.0] * 5, [0.2, -0.4, 1.3, -2.0, 0.0]])
    probability_from_logit = torch.sigmoid(noisy_or_logit(logits))
    assert torch.allclose(probability_from_logit, noisy_or_probability(logits), atol=1e-6)
    assert np.isfinite(probability_from_logit.numpy()).all()



def test_noisy_or_logit_gradient_is_finite_for_very_negative_logits():
    logits = torch.tensor([[-30.0] * 5, [-20.0] * 5, [-5.0] * 5, [8.0] * 5], requires_grad=True)
    noisy_or_logit(logits).sum().backward()
    assert torch.isfinite(logits.grad).all()
