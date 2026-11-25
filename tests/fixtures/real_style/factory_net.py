"""Factory functions returning configured networks (pytorch-cifar style)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock


class FactoryNet(nn.Module):
    """Network that receives its block class and depth list as constructor args."""

    def __init__(self, block, num_blocks, num_classes: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.blocks = nn.ModuleList(
            [block(8, 8) for _ in range(num_blocks[0])]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        return self.linear(torch.flatten(x, 1))


def SmallNet():
    return FactoryNet(ConvBlock, [1])


def DeepNet():
    return FactoryNet(ConvBlock, [2])
