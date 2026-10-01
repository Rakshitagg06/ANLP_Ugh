"""Loss and hierarchy aggregation functions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def noisy_or_probability(type_logits: torch.Tensor) -> torch.Tensor:
    """Compute ``1 - product(1 - sigmoid(logit_k))`` stably."""
    log_probability_none = F.logsigmoid(-type_logits).sum(dim=-1)
    return -torch.expm1(log_probability_none)


def noisy_or_logit(type_logits: torch.Tensor) -> torch.Tensor:
    """Return the exact logit corresponding to noisy-OR probabilities.

    If ``s = sum(softplus(z_k))``, the odds of noisy-OR are ``exp(s)-1``.
    This implementation evaluates ``log(expm1(s))`` without overflow.
    """
    s = F.softplus(type_logits).sum(dim=-1)
    split = math.log(2.0)
    small_branch = torch.log(torch.expm1(torch.clamp(s, min=1e-12, max=split)))
    # Clamp the unselected branch too: torch.where still backpropagates through
    # it, and log1p(-exp(-s)) is -inf (NaN gradient) when s underflows to ~0.
    s_large = torch.clamp(s, min=split)
    large_branch = s_large + torch.log1p(-torch.exp(-s_large))
    return torch.where(s <= split, small_branch, large_branch)


def masked_type_bce_loss(
    type_logits: torch.Tensor,
    type_targets: torch.Tensor,
    det_targets: torch.Tensor,
    positive_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate TYPE BCE only for examples where gold DET is positive."""
    mask = det_targets.bool()
    if not torch.any(mask):
        return type_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        type_logits[mask],
        type_targets[mask],
        pos_weight=positive_weights,
    )

