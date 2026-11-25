"""Model using an explicitly imported component class."""

from __future__ import annotations

import torch
import torch.nn as nn

from .components import SharedBlock


class SecondNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = SharedBlock(4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.pool(x)
        return self.head(torch.flatten(x, 1))
