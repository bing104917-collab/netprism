"""Utility helpers used by training scripts -- this file contains no model.

``FeatureCache`` subclasses ``nn.Module`` but has no ``forward`` method, so the
scanner must skip the whole file instead of reporting a model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters in ``module``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise each row of ``x``."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def top_k(logits: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the top-k values and indices along the last dimension."""
    return logits.topk(k, dim=-1)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Classification accuracy in percent."""
    predictions = logits.argmax(dim=-1)
    return float((predictions == targets).float().mean().item() * 100.0)


class FeatureCache(nn.Module):
    """Stores pre-computed features; deliberately has no ``forward``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("features", torch.zeros(0, dim))

    def add(self, features: torch.Tensor) -> None:
        self.features = torch.cat([self.features, features.detach()])

    def clear(self) -> None:
        self.features = torch.zeros_like(self.features)
