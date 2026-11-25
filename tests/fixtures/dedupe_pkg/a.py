"""First definition of DuplicateNet (kept as canonical by the scanner)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DuplicateNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(torch.flatten(x, 1))
