"""A plain multilayer perceptron for tabular / feature inputs.

Input : (1, 64) float32 feature vector
Output: (1, 10) class logits
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Three linear layers with GELU/ReLU activations and dropout."""

    def __init__(
        self,
        in_features: int = 64,
        hidden: int = 128,
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    model = MLP().eval()
    with torch.no_grad():
        print(model(torch.randn(1, 64)).shape)  # torch.Size([1, 10])
