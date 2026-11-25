"""Model file with a missing third-party import (dynamic import must fail soft)."""

from __future__ import annotations

import this_package_does_not_exist_netprism  # noqa: F401  (intentional)

import torch
import torch.nn as nn


class MissingDepNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
