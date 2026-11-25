"""A model that can only be built with an explicit ``width`` argument.

There is no default for ``width``, so NetPrism cannot instantiate it and must
fall back to static AST parsing.

Input : (1, 64) float32 feature vector
Output: (1, 10) class logits
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NeedsArgs(nn.Module):
    """MLP whose hidden width is a required constructor argument."""

    def __init__(self, width: int, num_classes: int = 10) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
