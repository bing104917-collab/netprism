"""Model built from wildcard-imported components (like Pytorch-UNet)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .components import *  # noqa: F401,F403  (wildcard import fixture)


class MainNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down = DownBlock(3, 8)
        self.out = OutBlock(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.down(x))
