"""String-configured model whose variant is only visible at a call site."""

from __future__ import annotations

import torch
import torch.nn as nn

_CONFIGS = {
    "tiny": 8,
    "wide": 32,
}


class StringNet(nn.Module):
    def __init__(self, variant: str) -> None:
        super().__init__()
        width = _CONFIGS.get(variant)
        if width is None:
            raise ValueError(f"unknown variant {variant!r}")
        self.embedding = nn.Embedding(16, width)
        self.head = nn.Linear(width, 4)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        return self.head(x.mean(dim=1))


def test():
    """A call site that reveals the concrete variant (like cifar's ``test()``)."""
    model = StringNet("tiny")
    return model(torch.zeros(1, 16, dtype=torch.long))


if __name__ == "__main__":
    test()
