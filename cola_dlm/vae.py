"""Utilities for the strictly causal text VAE."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from cola_dlm.config import VAEConfig
from cola_dlm.transformer import OutputProjection, TokenEmbedding, TransformerStack


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
    """Map token ids [batch, seq] to posterior tensors [batch, seq, latent]."""

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
        self.attention_pattern = config.attention_pattern

        _validate_text_vae_module_config(
            module_name="TextVAEEncoder",
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
            attention_pattern=self.attention_pattern,
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


class TextVAEDecoder(nn.Module):
    """Map [batch, seq] token ids and latents to [batch, seq, vocab] logits."""

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
            num_layers if num_layers is not None else config.decoder_layers
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
        self.attention_pattern = config.attention_pattern

        _validate_text_vae_module_config(
            module_name="TextVAEDecoder",
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
            attention_pattern=self.attention_pattern,
        )

        self.token_embedding = TokenEmbedding(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
        )
        self.latent_projection = nn.Linear(self.latent_dim, self.hidden_size)
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
        self.output_projection = OutputProjection(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        latents: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map [batch, seq] token ids and [batch, seq, latent] latents to logits."""

        _validate_decoder_inputs(
            token_ids=token_ids,
            latents=latents,
            latent_dim=self.latent_dim,
        )

        hidden_states = self.token_embedding(token_ids)
        hidden_states = hidden_states + self.latent_projection(latents)
        hidden_states = self.transformer(
            hidden_states,
            attention_mask=attention_mask,
            causal=True,
        )
        return self.output_projection(hidden_states)


@dataclass(frozen=True)
class TextVAEOutput:
    """VAE output: logits [batch, seq, vocab], latents [batch, seq, latent]."""

    logits: torch.Tensor
    posterior: DiagonalGaussianPosterior
    latents: torch.Tensor
    kl: torch.Tensor


class TextVAE(nn.Module):
    """Strictly causal text VAE wrapper with per-token latent tensors."""

    def __init__(self, config: VAEConfig | None = None) -> None:
        super().__init__()
        config = config or VAEConfig()
        self.encoder = TextVAEEncoder(config=config)
        self.decoder = TextVAEDecoder(config=config)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
        mask_loss_positions: torch.Tensor | None = None,
    ) -> TextVAEOutput:
        """Encode [batch, seq] tokens and decode logits [batch, seq, vocab]."""

        posterior = self.encoder(token_ids, attention_mask=attention_mask)
        latents = (
            posterior.mode()
            if deterministic
            else posterior.sample(generator=generator)
        )
        decoder_tokens = self.prepare_decoder_tokens(
            token_ids,
            mask_loss_positions=mask_loss_positions,
        )
        logits = self.decoder(
            decoder_tokens,
            latents,
            attention_mask=attention_mask,
        )
        return TextVAEOutput(
            logits=logits,
            posterior=posterior,
            latents=latents,
            kl=posterior.kl(),
        )

    def prepare_decoder_tokens(
        self,
        token_ids: torch.Tensor,
        *,
        mask_loss_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return decoder token ids, preserving [batch, seq] shape."""

        if (
            mask_loss_positions is not None
            and mask_loss_positions.shape != token_ids.shape
        ):
            raise ValueError("mask_loss_positions must match token_ids shape")
        return token_ids


def _validate_text_vae_module_config(
    *,
    module_name: str,
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
    attention_pattern: str,
) -> None:
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size!r}")
    if latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim!r}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size!r}")
    if patch_size != 1:
        raise NotImplementedError(f"{module_name} only supports patch_size=1")
    if attention_pattern != "causal":
        raise ValueError(
            f"{module_name} requires attention_pattern='causal', "
            f"got {attention_pattern!r}"
        )
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


def _validate_decoder_inputs(
    *,
    token_ids: torch.Tensor,
    latents: torch.Tensor,
    latent_dim: int,
) -> None:
    if token_ids.ndim != 2:
        raise ValueError("token_ids must be shaped [batch, seq]")
    if latents.ndim != 3:
        raise ValueError("latents must be shaped [batch, seq, latent_dim]")
    if token_ids.device != latents.device:
        raise ValueError("token_ids and latents must be on the same device")
    if not latents.is_floating_point():
        raise ValueError("latents must be a floating point tensor")
    if latents.shape[:2] != token_ids.shape:
        raise ValueError("token_ids and latents must share batch and seq dimensions")
    if latents.shape[-1] != latent_dim:
        raise ValueError(
            f"latents final dimension must equal latent_dim={latent_dim}, "
            f"got {latents.shape[-1]}"
        )


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
    "TextVAEDecoder",
    "TextVAEOutput",
    "TextVAE",
    "vae_logsnr",
)
