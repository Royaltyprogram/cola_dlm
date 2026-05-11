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


class RotaryEmbedding(nn.Module):
    """Apply rotary position embeddings to [batch, heads, seq, head_dim] tensors."""

    def __init__(self, base: float = 10000.0) -> None:
        super().__init__()
        self.base = base

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(query, key, position_ids)

        seq_len = query.shape[-2]
        head_dim = query.shape[-1]
        compute_dtype = _frequency_dtype(query, key)
        cos, sin = self._build_cos_sin(
            seq_len=seq_len,
            head_dim=head_dim,
            batch_size=query.shape[0],
            device=query.device,
            dtype=compute_dtype,
            position_ids=position_ids,
        )

        return _apply_rotary(query, cos, sin), _apply_rotary(key, cos, sin)

    def _build_cos_sin(
        self,
        seq_len: int,
        head_dim: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        position_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
        inv_freq = 1.0 / (self.base ** (steps / head_dim))

        if position_ids is None:
            positions = torch.arange(seq_len, device=device, dtype=dtype)
            angles = torch.outer(positions, inv_freq).unsqueeze(0).unsqueeze(0)
        else:
            positions = position_ids.to(device=device, dtype=dtype)
            if positions.ndim == 1:
                angles = (positions[:, None] * inv_freq).unsqueeze(0).unsqueeze(0)
            elif positions.ndim == 2:
                if positions.shape[0] != batch_size:
                    raise ValueError("position_ids batch size must match query batch size")
                angles = (positions[..., None] * inv_freq).unsqueeze(1)
            else:
                raise ValueError("position_ids must be shaped [seq] or [batch, seq]")

        return angles.cos(), angles.sin()

    @staticmethod
    def _validate_inputs(
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> None:
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError("query and key must be shaped [batch, heads, seq, head_dim]")
        if query.shape != key.shape:
            raise ValueError("query and key must have the same shape")
        if query.device != key.device:
            raise ValueError("query and key must be on the same device")
        if not query.is_floating_point() or not key.is_floating_point():
            raise ValueError("query and key must be floating point tensors")

        head_dim = query.shape[-1]
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")

        if position_ids is not None:
            if position_ids.ndim not in (1, 2):
                raise ValueError("position_ids must be shaped [seq] or [batch, seq]")
            if position_ids.shape[-1] != query.shape[-2]:
                raise ValueError("position_ids length must match query sequence length")


def _build_activation(activation: str) -> nn.Module:
    if activation == "gelu":
        return nn.GELU()
    if activation == "silu":
        return nn.SiLU()
    raise ValueError("activation must be 'gelu' or 'silu'")


def _frequency_dtype(query: torch.Tensor, key: torch.Tensor) -> torch.dtype:
    if query.dtype == torch.float64 or key.dtype == torch.float64:
        return torch.float64
    return torch.float32


def _apply_rotary(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.to(dtype=tensor.dtype)
    sin = sin.to(dtype=tensor.dtype)

    half_dim = tensor.shape[-1] // 2
    first_half = tensor[..., :half_dim]
    second_half = tensor[..., half_dim:]
    rotated = torch.cat(
        (
            first_half * cos - second_half * sin,
            second_half * cos + first_half * sin,
        ),
        dim=-1,
    )
    return rotated


__all__ = (
    "TokenEmbedding",
    "OutputProjection",
    "RMSNorm",
    "FeedForward",
    "RotaryEmbedding",
)
