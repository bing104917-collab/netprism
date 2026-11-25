"""A tiny GPT-style decoder with hand-written causal self-attention.

Input : (1, 16) int64 token ids in [0, vocab_size)
Output: (1, 16, 128) logits over the vocabulary (batch, seq, vocab_size)

The sequence length is fixed at 16 tokens, matching the causal mask size.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Multi-head causal self-attention (no nn.MultiheadAttention)."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, max_len: int = 16) -> None:
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        causal = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal.view(1, 1, max_len, max_len), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        # causal_mask marks future positions (upper triangle) as -inf.
        scores = scores.masked_fill(self.causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = self.attn_drop(weights) @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq, dim)
        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise FFN: expand 4x, GELU, project back."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LN block: attention and FFN, each wrapped in a residual connection."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TinyGPT(nn.Module):
    """Decoder-only language model: embeddings, N pre-LN blocks, LM head."""

    def __init__(
        self, vocab_size: int = 128, dim: int = 32, num_heads: int = 4,
        num_layers: int = 2, max_len: int = 16, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq = tokens.size(1)
        pos = torch.arange(seq, device=tokens.device)
        x = self.drop(self.token_emb(tokens) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
