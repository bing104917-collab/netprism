"""Second definition of DuplicateNet (dropped as a same-name duplicate)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DuplicateNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1)
        self.head = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.head(torch.flatten(x, 1))
