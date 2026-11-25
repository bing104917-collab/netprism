"""A tiny Transformer encoder that classifies a sequence of tokens.

Input : (1, 16) int64 token ids in [0, vocab_size)
Output: (1, 10) class logits

The sequence length is fixed at 16 tokens; the sinusoidal positional encoding
is a non-persistent buffer of exactly that length, not a module.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TinyTransformerClassifier(nn.Module):
    """Token embedding -> sinusoidal positions -> TransformerEncoder -> mean pool -> head."""

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 10,
        max_len: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoding", pe, persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(tokens) * math.sqrt(self.d_model)
        x = x + self.pos_encoding
        x = self.encoder(x)
        return self.head(self.norm(x.mean(dim=1)))
