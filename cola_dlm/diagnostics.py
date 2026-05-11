"""Reusable diagnostics for VAE and latent health checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cola_dlm.vae import TextVAEOutput, vae_logsnr


@dataclass(frozen=True)
class VAEDiagnostics:
    """Scalar VAE diagnostics suitable for flat metric logging."""

    reconstruction_accuracy: torch.Tensor
    logsnr: torch.Tensor
    latent_norm_mean: torch.Tensor
    latent_norm_std: torch.Tensor
    posterior_variance_mean: torch.Tensor
    posterior_variance_std: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return diagnostics with stable public names."""

        return {
            "reconstruction_accuracy": self.reconstruction_accuracy,
            "logsnr": self.logsnr,
            "latent_norm_mean": self.latent_norm_mean,
            "latent_norm_std": self.latent_norm_std,
            "posterior_variance_mean": self.posterior_variance_mean,
            "posterior_variance_std": self.posterior_variance_std,
        }


def compute_vae_diagnostics(
    output: TextVAEOutput,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> VAEDiagnostics:
    """Compute detached scalar diagnostics for a ``TextVAEOutput``."""

    _validate_diagnostic_inputs(output, token_ids, attention_mask)

    logits = output.logits.detach()
    token_ids = token_ids.detach()
    reconstruction_accuracy = _reconstruction_accuracy(
        logits,
        token_ids,
        attention_mask.detach() if attention_mask is not None else None,
    )

    mu = output.posterior.mu.detach()
    logvar = output.posterior.logvar.detach()
    latents = output.latents.detach()
    latent_norms = latents.norm(dim=-1)
    posterior_variance = torch.exp(logvar)

    return VAEDiagnostics(
        reconstruction_accuracy=reconstruction_accuracy.detach(),
        logsnr=vae_logsnr(mu, logvar).detach(),
        latent_norm_mean=latent_norms.mean().detach(),
        latent_norm_std=latent_norms.std(unbiased=False).detach(),
        posterior_variance_mean=posterior_variance.mean().detach(),
        posterior_variance_std=posterior_variance.std(unbiased=False).detach(),
    )


def _reconstruction_accuracy(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    correct = logits.argmax(dim=-1).eq(token_ids).to(dtype=logits.dtype)
    if attention_mask is None:
        return correct.mean()

    weights = attention_mask.to(dtype=logits.dtype)
    return (correct * weights).sum() / weights.sum()


def _validate_diagnostic_inputs(
    output: TextVAEOutput,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> None:
    if not isinstance(output, TextVAEOutput):
        raise TypeError("output must be a TextVAEOutput")
    if token_ids.ndim != 2:
        raise ValueError("token_ids must be shaped [batch, seq]")
    if output.logits.ndim != 3:
        raise ValueError("output.logits must be shaped [batch, seq, vocab]")
    if not output.logits.is_floating_point():
        raise ValueError("output.logits must be a floating point tensor")
    if token_ids.dtype != torch.long:
        raise ValueError("token_ids must be a torch.long tensor")
    if token_ids.numel() == 0:
        raise ValueError("token_ids must contain at least one token")
    if output.logits.shape[-1] <= 0:
        raise ValueError("output.logits must include at least one vocab entry")
    if torch.any(token_ids < 0).item():
        raise ValueError("token_ids must be non-negative")
    if torch.any(token_ids >= output.logits.shape[-1]).item():
        raise ValueError("token_ids must be less than logits vocab size")
    if output.logits.shape[:2] != token_ids.shape:
        raise ValueError("output.logits and token_ids must share batch and seq")
    if output.logits.device != token_ids.device:
        raise ValueError("output.logits and token_ids must be on the same device")
    if output.posterior.mu.ndim != 3:
        raise ValueError(
            "output.posterior tensors must be shaped [batch, seq, latent]"
        )
    if output.posterior.mu.device != output.logits.device:
        raise ValueError(
            "output.posterior.mu and output.logits must be on the same device"
        )
    if output.posterior.logvar.device != output.logits.device:
        raise ValueError(
            "output.posterior.logvar and output.logits must be on the same device"
        )
    if output.latents.shape != output.posterior.mu.shape:
        raise ValueError(
            "output.latents and posterior tensors must have the same shape"
        )
    if not output.latents.is_floating_point():
        raise ValueError("output.latents must be a floating point tensor")
    if output.latents.shape[:2] != token_ids.shape:
        raise ValueError("output.latents and token_ids must share batch and seq")
    if output.latents.device != output.logits.device:
        raise ValueError("output.latents and output.logits must be on the same device")
    if attention_mask is None:
        return
    if attention_mask.shape != token_ids.shape:
        raise ValueError("attention_mask must match token_ids shape")
    if attention_mask.device != token_ids.device:
        raise ValueError("attention_mask and token_ids must be on the same device")
    if attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be a boolean tensor")
    if not attention_mask.any().item():
        raise ValueError("attention_mask must select at least one token")


__all__ = ("VAEDiagnostics", "compute_vae_diagnostics")
