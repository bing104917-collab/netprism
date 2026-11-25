"""A miniature U-Net for binary segmentation.

Input : (1, 3, 32, 32) float32 image
Output: (1, 1, 32, 32) per-pixel logits
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) applied twice."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetMini(nn.Module):
    """Two-stage encoder/decoder with transposed-conv upsampling and skip cats."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base: int = 16) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)  # 32x32
        self.enc2 = DoubleConv(base, base * 2)  # 16x16
        self.enc3 = DoubleConv(base * 2, base * 4)  # 8x8 (bottleneck)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)  # cat: 32 + 32 channels
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)  # cat: 16 + 16 channels
        self.head = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)  # (B, 16, 32, 32)
        skip2 = self.enc2(self.pool(skip1))  # (B, 32, 16, 16)
        bottleneck = self.enc3(self.pool(skip2))  # (B, 64, 8, 8)
        x = self.up2(bottleneck)  # (B, 32, 16, 16)
        x = self.dec2(torch.cat([skip2, x], dim=1))
        x = self.up1(x)  # (B, 16, 32, 32)
        x = self.dec1(torch.cat([skip1, x], dim=1))
        return self.head(x)
