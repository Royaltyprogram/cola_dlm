"""Small transformer building blocks shared by Cola DLM models."""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Embed token ids shaped [batch, seq] into [batch, seq, hidden]."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class OutputProjection(nn.Module):
    """Project [batch, seq, hidden] hidden states to [batch, seq, vocab] logits."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, vocab_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states)


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the final tensor dimension."""

    def __init__(self, hidden_size: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return normalized * self.weight


class FeedForward(nn.Module):
    """Two-layer feed-forward block preserving [batch, seq, hidden] shape."""

    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, ffn_size)
        self.activation = _build_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(ffn_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_projection(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.output_projection(hidden_states)
        return self.dropout(hidden_states)


def _build_activation(activation: str) -> nn.Module:
    if activation == "gelu":
        return nn.GELU()
    if activation == "silu":
        return nn.SiLU()
    raise ValueError("activation must be 'gelu' or 'silu'")


__all__ = (
    "TokenEmbedding",
    "OutputProjection",
    "RMSNorm",
    "FeedForward",
)
