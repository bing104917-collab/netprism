"""A conv net whose forward pass contains data-dependent control flow.

The ``if x.mean() > 0`` branch depends on tensor *values*, so ``torch.fx``
symbolic tracing cannot follow it; NetPrism must fall back to an eager hook
trace (and note the unresolved branch).  Plain eager execution works fine.

Input : (1, 3, 32, 32) float32 RGB image
Output: (1, 10) class logits
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicNet(nn.Module):
    """Small CNN with a value-dependent two-way head."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.positive = nn.Linear(32, 32)
        self.negative = nn.Linear(32, 32)
        self.head = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.features(x), 1)
        # Data-dependent branch: this is what defeats symbolic tracing.
        if x.mean() > 0:
            x = torch.relu(self.positive(x))
        else:
            x = torch.relu(self.negative(x))
        return self.head(x)


if __name__ == "__main__":
    model = DynamicNet().eval()
    with torch.no_grad():
        print(model(torch.randn(1, 3, 32, 32)).shape)  # torch.Size([1, 10])
