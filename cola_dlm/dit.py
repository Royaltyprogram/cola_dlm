"""DiT backbone helpers for the Cola DLM reproduction."""

from __future__ import annotations

import math

import torch
from torch import nn

from cola_dlm.config import DiTConfig
from cola_dlm.transformer import RMSNorm, TransformerBlock


class TimestepEmbedding(nn.Module):
    """Embed normalized scalar timesteps into hidden-size conditioning vectors."""

    def __init__(self, hidden_size: int, max_period: float = 10000.0) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if max_period <= 0:
            raise ValueError("max_period must be positive")

        self.hidden_size = hidden_size
        self.max_period = max_period
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = self._validate_timesteps(timesteps)
        features = self._sinusoidal_features(timesteps)
        return self.projection(features)

    def _validate_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        if not isinstance(timesteps, torch.Tensor):
            raise TypeError("timesteps must be a torch.Tensor")
        if not timesteps.is_floating_point():
            raise ValueError("timesteps must be a floating point tensor")
        if timesteps.ndim == 1:
            return timesteps
        if timesteps.ndim == 2 and timesteps.shape[1] == 1:
            return timesteps[:, 0]
        raise ValueError("timesteps must be shaped [batch] or [batch, 1]")

    def _sinusoidal_features(self, timesteps: torch.Tensor) -> torch.Tensor:
        projection_dtype = self.projection[0].weight.dtype
        compute_dtype = (
            torch.float64 if projection_dtype == torch.float64 else torch.float32
        )
        timesteps = timesteps.to(dtype=compute_dtype)

        half_size = (self.hidden_size + 1) // 2
        steps = torch.arange(half_size, device=timesteps.device, dtype=compute_dtype)
        frequencies = torch.exp(-math.log(self.max_period) * steps / half_size)
        angles = timesteps[:, None] * frequencies[None, :]
        features = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return features[:, : self.hidden_size].to(dtype=projection_dtype)


class BlockCausalTextDiT(nn.Module):
    """Block-causal DiT backbone over packed latent sequences."""

    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.latent_dim, config.hidden_size)
        self.timestep_embedding = TimestepEmbedding(config.hidden_size)
        use_rope = config.positional_encoding == "rope"
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    ffn_size=config.ffn_size,
                    head_dim=config.attention_head_dim,
                    dropout=config.dropout,
                    activation=config.activation,
                    use_rope=use_rope,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_norm = RMSNorm(config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, config.latent_dim)

    def forward(
        self,
        packed_latents: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del segment_ids
        self._validate_packed_latents(packed_latents)
        self._validate_attention_mask(attention_mask, packed_latents)
        if (
            isinstance(timesteps, torch.Tensor)
            and timesteps.device != packed_latents.device
        ):
            raise ValueError("timesteps must be on the same device as packed_latents")

        hidden_states = self.input_projection(packed_latents)
        time_embedding = self.timestep_embedding(timesteps)
        if time_embedding.shape[0] != packed_latents.shape[0]:
            raise ValueError(
                "timesteps batch size must match packed_latents batch size"
            )
        time_embedding = time_embedding.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states + time_embedding[:, None, :],
                attention_mask=attention_mask,
                causal=False,
            )

        hidden_states = self.output_norm(hidden_states)
        return self.output_projection(hidden_states)

    def _validate_packed_latents(self, packed_latents: torch.Tensor) -> None:
        if not isinstance(packed_latents, torch.Tensor):
            raise TypeError("packed_latents must be a torch.Tensor")
        if packed_latents.ndim != 3:
            raise ValueError(
                "packed_latents must be shaped [batch, packed_len, latent_dim]"
            )
        if not packed_latents.is_floating_point():
            raise ValueError("packed_latents must be a floating point tensor")
        if packed_latents.shape[-1] != self.config.latent_dim:
            raise ValueError(
                "packed_latents latent dimension must match config.latent_dim "
                f"(got {packed_latents.shape[-1]}, expected {self.config.latent_dim})"
            )

    @staticmethod
    def _validate_attention_mask(
        attention_mask: torch.Tensor,
        packed_latents: torch.Tensor,
    ) -> None:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch.Tensor")

        packed_len = packed_latents.shape[1]
        if attention_mask.ndim == 2:
            expected = (packed_len, packed_len)
            if attention_mask.shape != expected:
                raise ValueError(
                    "2D attention_mask length must match packed_latents packed_len "
                    f"(got {tuple(attention_mask.shape)}, expected {expected})"
                )
        elif attention_mask.ndim in (3, 4):
            if attention_mask.shape[-2:] != (packed_len, packed_len):
                raise ValueError(
                    "attention_mask sequence dimensions must match "
                    "packed_latents packed_len "
                    f"(got {tuple(attention_mask.shape[-2:])}, "
                    f"expected {(packed_len, packed_len)})"
                )
            if attention_mask.shape[0] not in (1, packed_latents.shape[0]):
                raise ValueError(
                    "attention_mask batch dimension must be 1 or match "
                    "packed_latents batch size"
                )
        else:
            raise ValueError(
                "attention_mask must be shaped [packed_len, packed_len], "
                "[batch, packed_len, packed_len], or "
                "[batch, heads, packed_len, packed_len]"
            )
