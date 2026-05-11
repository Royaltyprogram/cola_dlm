"""Utilities for the causal text VAE."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from cola_dlm.config import VAEConfig
from cola_dlm.transformer import TokenEmbedding, TransformerStack


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


class TextVAEEncoder(nn.Module):
    """Causal token encoder returning per-token posterior parameters."""

    def __init__(
        self,
        config: VAEConfig | None = None,
        *,
        vocab_size: int | None = None,
        latent_dim: int | None = None,
        patch_size: int | None = None,
        num_layers: int | None = None,
        hidden_size: int | None = None,
        ffn_size: int | None = None,
        num_attention_heads: int | None = None,
        attention_head_dim: int | None = None,
        dropout: float | None = None,
        activation: str | None = None,
        use_rope: bool | None = None,
    ) -> None:
        super().__init__()
        config = config or VAEConfig()

        self.vocab_size = vocab_size if vocab_size is not None else config.vocab_size
        self.latent_dim = latent_dim if latent_dim is not None else config.latent_dim
        self.patch_size = patch_size if patch_size is not None else config.patch_size
        self.hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        self.num_layers = (
            num_layers if num_layers is not None else config.encoder_layers
        )
        self.ffn_size = ffn_size if ffn_size is not None else config.ffn_size
        self.num_attention_heads = (
            num_attention_heads
            if num_attention_heads is not None
            else config.num_attention_heads
        )
        if attention_head_dim is not None:
            self.attention_head_dim = attention_head_dim
        elif hidden_size is not None or num_attention_heads is not None:
            self.attention_head_dim = (
                self.hidden_size // self.num_attention_heads
                if self.num_attention_heads > 0
                else config.attention_head_dim
            )
        else:
            self.attention_head_dim = config.attention_head_dim
        self.dropout = dropout if dropout is not None else config.dropout
        self.activation = activation if activation is not None else config.activation
        self.use_rope = use_rope if use_rope is not None else config.use_rope

        _validate_text_encoder_config(
            vocab_size=self.vocab_size,
            latent_dim=self.latent_dim,
            patch_size=self.patch_size,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            ffn_size=self.ffn_size,
            num_attention_heads=self.num_attention_heads,
            attention_head_dim=self.attention_head_dim,
            dropout=self.dropout,
            activation=self.activation,
            use_rope=self.use_rope,
        )

        self.token_embedding = TokenEmbedding(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
        )
        self.transformer = TransformerStack(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            ffn_size=self.ffn_size,
            head_dim=self.attention_head_dim,
            dropout=self.dropout,
            activation=self.activation,
            use_rope=self.use_rope,
        )
        self.posterior_projection = nn.Linear(self.hidden_size, 2 * self.latent_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> DiagonalGaussianPosterior:
        """Map [batch, seq] token ids to [batch, seq, latent] posterior tensors."""

        if token_ids.ndim != 2:
            raise ValueError("token_ids must be shaped [batch, seq]")

        hidden_states = self.token_embedding(token_ids)
        hidden_states = self.transformer(
            hidden_states,
            attention_mask=attention_mask,
            causal=True,
        )
        mu, logvar = self.posterior_projection(hidden_states).chunk(2, dim=-1)
        return DiagonalGaussianPosterior(mu=mu, logvar=logvar)


def _validate_text_encoder_config(
    *,
    vocab_size: int,
    latent_dim: int,
    patch_size: int,
    num_layers: int,
    hidden_size: int,
    ffn_size: int,
    num_attention_heads: int,
    attention_head_dim: int,
    dropout: float,
    activation: str,
    use_rope: bool,
) -> None:
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size!r}")
    if latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim!r}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size!r}")
    if patch_size != 1:
        raise NotImplementedError("TextVAEEncoder only supports patch_size=1")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers!r}")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size!r}")
    if ffn_size <= 0:
        raise ValueError(f"ffn_size must be positive, got {ffn_size!r}")
    if num_attention_heads <= 0:
        raise ValueError(
            f"num_attention_heads must be positive, got {num_attention_heads!r}"
        )
    if attention_head_dim <= 0:
        raise ValueError(
            f"attention_head_dim must be positive, got {attention_head_dim!r}"
        )
    if hidden_size != num_attention_heads * attention_head_dim:
        raise ValueError(
            "hidden_size must equal num_attention_heads * attention_head_dim"
        )
    if dropout < 0 or dropout >= 1:
        raise ValueError("dropout must be in the range [0, 1)")
    if activation not in ("gelu", "silu"):
        raise ValueError("activation must be 'gelu' or 'silu'")
    if use_rope and attention_head_dim % 2 != 0:
        raise ValueError("attention_head_dim must be even when use_rope=True")


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
    "TextVAEEncoder",
    "vae_logsnr",
)
