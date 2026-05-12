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


class MultiHeadAttention(nn.Module):
    """Self-attend [batch, seq, hidden] states with causal or custom masks."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if head_dim is None:
            if hidden_size % num_heads != 0:
                raise ValueError(
                    "hidden_size must be divisible by num_heads when head_dim is not set"
                )
            head_dim = hidden_size // num_heads
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if hidden_size != num_heads * head_dim:
            raise ValueError(
                "hidden_size must equal num_heads * head_dim "
                f"(got hidden_size={hidden_size}, num_heads={num_heads}, "
                f"head_dim={head_dim})"
            )
        if use_rope and head_dim % 2 != 0:
            raise ValueError("head_dim must be even when use_rope=True")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding() if use_rope else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must be shaped [batch, seq, hidden]")

        batch_size, seq_len, _ = hidden_states.shape
        query = self._split_heads(self.query_projection(hidden_states))
        key = self._split_heads(self.key_projection(hidden_states))
        value = self._split_heads(self.value_projection(hidden_states))

        if self.rotary is not None:
            query, key = self.rotary(query, key)

        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim**-0.5)
        if causal:
            causal_mask = torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=hidden_states.device,
            ).tril()
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)

        normalized_mask = _normalize_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            num_heads=self.num_heads,
            seq_len=seq_len,
            dtype=scores.dtype,
            device=scores.device,
        )
        if normalized_mask is not None:
            scores = scores + normalized_mask

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended = torch.matmul(attention_weights, value)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.hidden_size)
        return self.output_projection(attended)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block preserving [batch, seq, hidden] shape."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            use_rope=use_rope,
        )
        self.norm2 = RMSNorm(hidden_size)
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            ffn_size=ffn_size,
            dropout=dropout,
            activation=activation,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            causal=causal,
        )
        return hidden_states + self.feed_forward(self.norm2(hidden_states))


class TransformerStack(nn.Module):
    """Stack pre-norm blocks while preserving [batch, seq, hidden] shape."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ffn_size=ffn_size,
                    head_dim=head_dim,
                    dropout=dropout,
                    activation=activation,
                    use_rope=use_rope,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                causal=causal,
            )
        return hidden_states


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


def _normalize_attention_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Convert supported bool or additive masks to an additive score mask."""

    if attention_mask is None:
        return None

    is_boolean_mask = attention_mask.dtype == torch.bool
    if not is_boolean_mask and not attention_mask.is_floating_point():
        raise ValueError("attention_mask must be a bool or floating point tensor")

    if attention_mask.ndim == 2:
        if attention_mask.shape == (seq_len, seq_len):
            mask = attention_mask.unsqueeze(0).unsqueeze(0)
        elif is_boolean_mask and attention_mask.shape == (batch_size, seq_len):
            mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("2D attention_mask must be shaped [seq, seq]")
    elif attention_mask.ndim == 3:
        if attention_mask.shape[-2:] != (seq_len, seq_len):
            raise ValueError("3D attention_mask must be shaped [batch, seq, seq]")
        if attention_mask.shape[0] not in (1, batch_size):
            raise ValueError("attention_mask batch dimension must be 1 or batch size")
        mask = attention_mask.unsqueeze(1)
    elif attention_mask.ndim == 4:
        if attention_mask.shape[-2:] != (seq_len, seq_len):
            raise ValueError(
                "4D attention_mask must be shaped [batch, heads, seq, seq]"
            )
        if attention_mask.shape[0] not in (1, batch_size):
            raise ValueError("attention_mask batch dimension must be 1 or batch size")
        if attention_mask.shape[1] not in (1, num_heads):
            raise ValueError("attention_mask head dimension must be 1 or num_heads")
        mask = attention_mask
    else:
        raise ValueError(
            "attention_mask must be shaped [seq, seq], [batch, seq, seq], "
            "or [batch, heads, seq, seq]"
        )

    mask = mask.to(device=device)
    if is_boolean_mask:
        additive_mask = torch.zeros(mask.shape, dtype=dtype, device=device)
        return additive_mask.masked_fill(~mask, torch.finfo(dtype).min)
    return mask.to(dtype=dtype)


__all__ = (
    "TokenEmbedding",
    "OutputProjection",
    "RMSNorm",
    "FeedForward",
    "RotaryEmbedding",
    "MultiHeadAttention",
    "TransformerBlock",
    "TransformerStack",
)
