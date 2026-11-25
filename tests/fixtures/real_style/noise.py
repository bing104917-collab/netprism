"""Blocks and wrapper classes that must not be reported as top-level models."""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyAttention(nn.Module):
    """Helper referenced (and instantiated) by :class:`FineNet`."""

    def __init__(self, dim: int = 16, heads: int = 2) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        weights = torch.softmax(q @ k.transpose(-2, -1), dim=-1)
        return self.proj((weights @ v).transpose(1, 2).reshape(batch, seq, dim))


class TrainerLike(nn.Module):
    """Wrapper that requires another model: not a standalone architecture."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ModelWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FineNet(nn.Module):
    """Real top-level model; owns the helper attention block."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, dim)
        self.blocks = nn.Sequential(TinyAttention(dim), TinyAttention(dim))
        self.head = nn.Linear(dim, 4)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        x = self.blocks(x)
        return self.head(x.mean(dim=1))
