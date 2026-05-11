"""Stage 1 Text VAE training helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cola_dlm.vae import TextVAEOutput, vae_logsnr


@dataclass(frozen=True)
class Stage1VAELoss:
    """Structured scalar losses and diagnostics for Stage 1 VAE training."""

    loss: torch.Tensor
    reconstruction_nll: torch.Tensor
    kl: torch.Tensor
    mask_loss: torch.Tensor
    logsnr: torch.Tensor


def compute_stage1_vae_loss(
    output: TextVAEOutput,
    token_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    mask_labels: torch.Tensor | None = None,
    lambda_kl: float = 1.0,
    lambda_mask: float = 0.0,
) -> Stage1VAELoss:
    """Compute reconstruction and KL loss terms for a TextVAE forward pass."""

    _validate_non_negative_weight("lambda_kl", lambda_kl)
    _validate_non_negative_weight("lambda_mask", lambda_mask)
    _validate_loss_shapes(output=output, token_ids=token_ids)
    if attention_mask is not None:
        _validate_attention_mask(attention_mask, token_ids)

    reconstruction_nll = _masked_mean(
        _per_token_cross_entropy(output.logits, token_ids),
        attention_mask,
    )
    kl = _masked_mean(output.kl, attention_mask)
    mask_loss = _zero_scalar_like(output.logits)
    if mask_labels is not None and lambda_mask != 0.0:
        raise NotImplementedError("non-zero mask loss is added in the masking step")
    logsnr = vae_logsnr(output.posterior.mu, output.posterior.logvar)
    loss = reconstruction_nll + lambda_kl * kl + lambda_mask * mask_loss
    return Stage1VAELoss(
        loss=loss,
        reconstruction_nll=reconstruction_nll,
        kl=kl,
        mask_loss=mask_loss,
        logsnr=logsnr,
    )


def _per_token_cross_entropy(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        token_ids.reshape(-1),
        reduction="none",
    ).reshape(token_ids.shape)


def _masked_mean(
    values: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is None:
        return values.mean()

    weights = attention_mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum()


def _zero_scalar_like(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.new_zeros(())


def _validate_loss_shapes(output: TextVAEOutput, token_ids: torch.Tensor) -> None:
    if output.logits.ndim != 3:
        raise ValueError("output.logits must be shaped [batch, seq, vocab]")
    if token_ids.ndim != 2:
        raise ValueError("token_ids must be shaped [batch, seq]")
    if output.logits.shape[:2] != token_ids.shape:
        raise ValueError("output.logits and token_ids must share batch and seq shape")
    if output.kl.shape != token_ids.shape:
        raise ValueError("output.kl must be shaped [batch, seq]")
    if output.logits.device != token_ids.device:
        raise ValueError("output.logits and token_ids must be on the same device")
    if output.kl.device != token_ids.device:
        raise ValueError("output.kl and token_ids must be on the same device")


def _validate_attention_mask(
    attention_mask: torch.Tensor,
    token_ids: torch.Tensor,
) -> None:
    if attention_mask.shape != token_ids.shape:
        raise ValueError("attention_mask must match token_ids shape")
    if attention_mask.device != token_ids.device:
        raise ValueError("attention_mask and token_ids must be on the same device")
    if attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be a boolean tensor")
    if not attention_mask.any().item():
        raise ValueError("attention_mask must select at least one token")


def _validate_non_negative_weight(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


__all__ = ("Stage1VAELoss", "compute_stage1_vae_loss")
