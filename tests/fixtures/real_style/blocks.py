"""Building blocks reused by the fixture models (not standalone models)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """A small conv-norm-activation block used as a constructor argument."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Attention(nn.Module):
    """Attention helper referenced by :class:`AttentionNet`."""

    def __init__(self, dim: int = 32, heads: int = 4) -> None:
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
        out = (weights @ v).transpose(1, 2).reshape(batch, seq, dim)
        return self.proj(out)
