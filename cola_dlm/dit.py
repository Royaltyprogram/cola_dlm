"""DiT backbone helpers for the Cola DLM reproduction."""

from __future__ import annotations

import math

import torch
from torch import nn


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
