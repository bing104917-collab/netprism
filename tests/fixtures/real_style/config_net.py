"""Model that needs a configuration object (nanoGPT ``GPT(config)`` style)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import FixtureConfig


class ConfigNet(nn.Module):
    def __init__(self, config: FixtureConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 2,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        x = self.encoder(x)
        return self.head(self.norm(x.mean(dim=1)))
