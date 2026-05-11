"""Utilities for the causal text VAE."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class DiagonalGaussianPosterior:
    """Diagonal Gaussian posterior with tensors shaped [..., latent_dim]."""

    mu: torch.Tensor
    logvar: torch.Tensor

    def __post_init__(self) -> None:
        _validate_matching_floating_tensors(self.mu, self.logvar, "mu", "logvar")

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample with the reparameterization trick, preserving posterior shape."""

        noise = torch.randn(
            self.mu.shape,
            generator=generator,
            device=self.mu.device,
            dtype=self.mu.dtype,
        )
        return self.mu + noise * torch.exp(0.5 * self.logvar)

    def mode(self) -> torch.Tensor:
        """Return the posterior mean."""

        return self.mu

    def kl(self) -> torch.Tensor:
        """KL divergence to a standard normal, reduced over latent_dim."""

        per_dim_kl = torch.exp(self.logvar) + self.mu.pow(2) - 1.0 - self.logvar
        return 0.5 * per_dim_kl.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Differential entropy, reduced over latent_dim."""

        per_dim_entropy = 0.5 * (1.0 + math.log(2.0 * math.pi) + self.logvar)
        return per_dim_entropy.sum(dim=-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log probability of value, reduced over latent_dim."""

        _validate_matching_floating_tensors(value, self.mu, "value", "mu")
        squared_error = (value - self.mu).pow(2)
        per_dim_log_prob = -0.5 * (
            math.log(2.0 * math.pi)
            + self.logvar
            + squared_error * torch.exp(-self.logvar)
        )
        return per_dim_log_prob.sum(dim=-1)


def vae_logsnr(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Return empirical log(mean(mu**2) / mean(exp(logvar))) as a scalar tensor."""

    _validate_matching_floating_tensors(mu, logvar, "mu", "logvar")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps!r}")

    signal_power = mu.pow(2).mean().clamp_min(eps)
    noise_power = torch.exp(logvar).mean().clamp_min(eps)
    return torch.log(signal_power) - torch.log(noise_power)


def _validate_matching_floating_tensors(
    left: torch.Tensor,
    right: torch.Tensor,
    left_name: str,
    right_name: str,
) -> None:
    if left.shape != right.shape:
        raise ValueError(
            f"{left_name} and {right_name} must have the same shape, "
            f"got {tuple(left.shape)} and {tuple(right.shape)}"
        )
    if left.device != right.device:
        raise ValueError(f"{left_name} and {right_name} must be on the same device")
    if not left.is_floating_point() or not right.is_floating_point():
        raise ValueError(f"{left_name} and {right_name} must be floating point tensors")
    if left.ndim == 0:
        raise ValueError(
            f"{left_name} and {right_name} must include a latent dimension"
        )


__all__ = (
    "DiagonalGaussianPosterior",
    "vae_logsnr",
)
