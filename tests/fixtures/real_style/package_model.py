"""Model defined in a package module that uses package-relative imports."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import Attention
from .config import FixtureConfig


class PackageNet(nn.Module):
    """Uses relative imports (``from .blocks import Attention``) like real repos."""

    def __init__(self, config: FixtureConfig, dim: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, dim)
        self.attn = Attention(dim=dim, heads=2)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, config.num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        x = self.attn(x)
        return self.head(self.norm(x.mean(dim=1)))
