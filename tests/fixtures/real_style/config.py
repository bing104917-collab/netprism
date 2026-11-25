"""Dataclass configuration with declared defaults (nanoGPT style)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FixtureConfig:
    vocab_size: int = 32
    embed_dim: int = 8
    num_heads: int = 2
    depth: int = 2
    seq_len: int = 16
    num_classes: int = 4
    dropout: float = 0.0
