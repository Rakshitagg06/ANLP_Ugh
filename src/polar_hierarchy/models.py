"""M1, M2, and M4-core model implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from .losses import masked_type_bce_loss, noisy_or_logit, noisy_or_probability


@dataclass
class ModelOutput:
    loss: torch.Tensor | None
    det_logits: torch.Tensor | None
    type_logits: torch.Tensor | None
    loss_det: torch.Tensor | None = None
    loss_type: torch.Tensor | None = None


class EncoderWithPooling(nn.Module):
    def __init__(self, pretrained_name: str, revision: str | None, dropout: float) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_name, revision=revision)
        self.encoder = AutoModel.from_pretrained(pretrained_name, revision=revision, config=config)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = int(config.hidden_size)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.dropout(pooled)

    def encoder_parameters(self):
        return self.encoder.parameters()


class M1DetModel(EncoderWithPooling):
    def __init__(self, pretrained_name: str, revision: str | None, dropout: float) -> None:
        super().__init__(pretrained_name, revision, dropout)
        self.det_head = nn.Linear(self.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        det_labels: torch.Tensor | None = None,
        type_labels: torch.Tensor | None = None,  # unused; accepted so batches match other models
        **kwargs: Any,
    ) -> ModelOutput:
        det_logits = self.det_head(self.encode(input_ids, attention_mask, **kwargs)).squeeze(-1)
        loss = None if det_labels is None else F.binary_cross_entropy_with_logits(det_logits, det_labels)
        return ModelOutput(loss=loss, det_logits=det_logits, type_logits=None, loss_det=loss)


class M1TypeModel(EncoderWithPooling):
    def __init__(
        self,
        pretrained_name: str,
        revision: str | None,
        dropout: float,
        positive_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__(pretrained_name, revision, dropout)
        self.type_head = nn.Linear(self.hidden_size, 5)
        self.register_buffer("positive_weights", positive_weights, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        det_labels: torch.Tensor | None = None,
        type_labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        type_logits = self.type_head(self.encode(input_ids, attention_mask, **kwargs))
        loss = None
        if det_labels is not None and type_labels is not None:
            loss = masked_type_bce_loss(
                type_logits, type_labels, det_labels, self.positive_weights
            )
        return ModelOutput(loss=loss, det_logits=None, type_logits=type_logits, loss_type=loss)


class M2MultiTaskModel(EncoderWithPooling):
    def __init__(
        self,
        pretrained_name: str,
        revision: str | None,
        dropout: float,
        lambda_type: float,
        positive_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__(pretrained_name, revision, dropout)
        self.det_head = nn.Linear(self.hidden_size, 1)
        self.type_head = nn.Linear(self.hidden_size, 5)
        self.lambda_type = lambda_type
        self.register_buffer("positive_weights", positive_weights, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        det_labels: torch.Tensor | None = None,
        type_labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        pooled = self.encode(input_ids, attention_mask, **kwargs)
        det_logits = self.det_head(pooled).squeeze(-1)
        type_logits = self.type_head(pooled)
        if det_labels is None or type_labels is None:
            return ModelOutput(None, det_logits, type_logits)
        loss_det = F.binary_cross_entropy_with_logits(det_logits, det_labels)
        loss_type = masked_type_bce_loss(
            type_logits, type_labels, det_labels, self.positive_weights
        )
        loss = loss_det + self.lambda_type * loss_type
        return ModelOutput(loss, det_logits, type_logits, loss_det, loss_type)


class M4CoreModel(EncoderWithPooling):
    def __init__(
        self,
        pretrained_name: str,
        revision: str | None,
        dropout: float,
        lambda_type: float,
        positive_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__(pretrained_name, revision, dropout)
        self.type_head = nn.Linear(self.hidden_size, 5)
        self.lambda_type = lambda_type
        self.register_buffer("positive_weights", positive_weights, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        det_labels: torch.Tensor | None = None,
        type_labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        type_logits = self.type_head(self.encode(input_ids, attention_mask, **kwargs))
        det_logits = noisy_or_logit(type_logits)
        if det_labels is None or type_labels is None:
            return ModelOutput(None, det_logits, type_logits)
        loss_det = F.binary_cross_entropy_with_logits(det_logits, det_labels)
        loss_type = masked_type_bce_loss(
            type_logits, type_labels, det_labels, self.positive_weights
        )
        loss = loss_det + self.lambda_type * loss_type
        return ModelOutput(loss, det_logits, type_logits, loss_det, loss_type)

    @staticmethod
    def detection_probability(type_logits: torch.Tensor) -> torch.Tensor:
        return noisy_or_probability(type_logits)


def build_model(config: Dict[str, Any], positive_weights: torch.Tensor | None = None) -> nn.Module:
    model_type = config["experiment"]["model_type"]
    model_config = config["model"]
    common = {
        "pretrained_name": model_config["pretrained_name"],
        "revision": model_config.get("revision"),
        "dropout": float(model_config.get("dropout", 0.1)),
    }
    if model_type == "m1_det":
        return M1DetModel(**common)
    if model_type == "m1_type":
        return M1TypeModel(**common, positive_weights=positive_weights)
    if model_type == "m2":
        return M2MultiTaskModel(
            **common,
            lambda_type=float(model_config.get("lambda_type", 1.0)),
            positive_weights=positive_weights,
        )
    if model_type == "m4_core":
        return M4CoreModel(
            **common,
            lambda_type=float(model_config.get("lambda_type", 1.0)),
            positive_weights=positive_weights,
        )
    raise ValueError(f"Unknown model_type: {model_type}")
