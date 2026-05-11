"""Stage 1 Text VAE training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import torch
import torch.nn.functional as F

from cola_dlm.config import Stage1Config
from cola_dlm.vae import TextVAE, TextVAEOutput, vae_logsnr


_DEFAULT_MASK_IGNORE_INDEX = -100


@dataclass(frozen=True)
class Stage1VAELoss:
    """Structured scalar losses and diagnostics for Stage 1 VAE training."""

    loss: torch.Tensor
    reconstruction_nll: torch.Tensor
    kl: torch.Tensor
    mask_loss: torch.Tensor
    logsnr: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return diagnostics with stable public names."""

        return {
            "loss": self.loss,
            "reconstruction_nll": self.reconstruction_nll,
            "kl": self.kl,
            "mask_loss": self.mask_loss,
            "logsnr": self.logsnr,
        }


@dataclass(frozen=True)
class Stage1MaskingPolicy:
    """Simple token masking policy for optional Stage 1 mask loss."""

    mask_token_id: int
    mask_probability: float
    ignore_index: int = _DEFAULT_MASK_IGNORE_INDEX

    def __post_init__(self) -> None:
        _validate_mask_token_id(self.mask_token_id)
        _validate_mask_probability(self.mask_probability)
        _validate_ignore_index(self.ignore_index)


def apply_stage1_masking(
    token_ids: torch.Tensor,
    policy: Stage1MaskingPolicy,
    *,
    attention_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return masked decoder tokens, mask labels, and selected positions."""

    _validate_token_ids_for_masking(token_ids)
    if attention_mask is not None:
        _validate_attention_mask(attention_mask, token_ids, require_any=False)

    sampled_positions = (
        torch.rand(
            token_ids.shape,
            generator=generator,
            device=token_ids.device,
        )
        < policy.mask_probability
    )
    valid_positions = (
        torch.ones_like(sampled_positions, dtype=torch.bool)
        if attention_mask is None
        else attention_mask
    )
    mask_positions = sampled_positions & valid_positions

    masked_decoder_inputs = token_ids.clone()
    masked_decoder_inputs[mask_positions] = policy.mask_token_id

    mask_labels = torch.full_like(token_ids, policy.ignore_index)
    mask_labels[mask_positions] = token_ids[mask_positions]
    return masked_decoder_inputs, mask_labels, mask_positions


def compute_stage1_vae_loss(
    output: TextVAEOutput,
    token_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    mask_labels: torch.Tensor | None = None,
    stage1_config: Stage1Config | None = None,
    lambda_kl: float | None = None,
    lambda_mask: float | None = None,
    mask_ignore_index: int = _DEFAULT_MASK_IGNORE_INDEX,
) -> Stage1VAELoss:
    """Compute reconstruction and KL loss terms for a TextVAE forward pass."""

    lambda_kl, lambda_mask = _resolve_stage1_weights(
        stage1_config=stage1_config,
        lambda_kl=lambda_kl,
        lambda_mask=lambda_mask,
    )
    _validate_ignore_index(mask_ignore_index)
    _validate_loss_shapes(output=output, token_ids=token_ids)
    if attention_mask is not None:
        _validate_attention_mask(attention_mask, token_ids, require_any=True)
    if mask_labels is not None:
        _validate_mask_labels(mask_labels, token_ids)

    reconstruction_nll = _masked_mean(
        _per_token_cross_entropy(output.logits, token_ids),
        attention_mask,
    )
    kl = _masked_mean(output.kl, attention_mask)
    mask_loss = _zero_scalar_like(output.logits)
    if mask_labels is not None and lambda_mask != 0.0:
        mask_loss = _masked_cross_entropy(
            output.logits,
            mask_labels,
            ignore_index=mask_ignore_index,
            attention_mask=attention_mask,
        )
    logsnr = vae_logsnr(output.posterior.mu, output.posterior.logvar)
    loss = reconstruction_nll + lambda_kl * kl + lambda_mask * mask_loss
    return Stage1VAELoss(
        loss=loss,
        reconstruction_nll=reconstruction_nll,
        kl=kl,
        mask_loss=mask_loss,
        logsnr=logsnr,
    )


def stage1_pretraining_step(
    model: TextVAE,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    masking_policy: Stage1MaskingPolicy | None = None,
    stage1_config: Stage1Config | None = None,
    lambda_kl: float | None = None,
    lambda_mask: float | None = None,
    max_grad_norm: float | None = None,
    deterministic: bool = False,
    generator: torch.Generator | None = None,
) -> Stage1VAELoss:
    """Run one optimizer step for a tiny Stage 1 TextVAE pretraining batch."""

    lambda_kl, lambda_mask = _resolve_stage1_weights(
        stage1_config=stage1_config,
        lambda_kl=lambda_kl,
        lambda_mask=lambda_mask,
    )
    if max_grad_norm is not None:
        _validate_non_negative_weight("max_grad_norm", max_grad_norm)

    decoder_token_ids = None
    mask_labels = None
    mask_positions = None
    mask_ignore_index = _DEFAULT_MASK_IGNORE_INDEX
    if masking_policy is not None and lambda_mask != 0.0:
        decoder_token_ids, mask_labels, mask_positions = apply_stage1_masking(
            token_ids,
            masking_policy,
            attention_mask=attention_mask,
            generator=generator,
        )
        mask_ignore_index = masking_policy.ignore_index

    model.train()
    output = model(
        token_ids,
        attention_mask=attention_mask,
        deterministic=deterministic,
        generator=generator,
        decoder_token_ids=decoder_token_ids,
        mask_loss_positions=mask_positions,
    )
    loss = compute_stage1_vae_loss(
        output,
        token_ids,
        attention_mask=attention_mask,
        mask_labels=mask_labels,
        lambda_kl=lambda_kl,
        lambda_mask=lambda_mask,
        mask_ignore_index=mask_ignore_index,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss


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


def _masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    selected_positions = labels != ignore_index
    if attention_mask is not None:
        selected_positions = selected_positions & attention_mask
    if not selected_positions.any().item():
        return _zero_scalar_like(logits)
    return F.cross_entropy(logits[selected_positions], labels[selected_positions])


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


def _resolve_stage1_weights(
    *,
    stage1_config: Stage1Config | None,
    lambda_kl: float | None,
    lambda_mask: float | None,
) -> tuple[float, float]:
    config_kl = None if stage1_config is None else stage1_config.kl_weight
    config_mask = None if stage1_config is None else stage1_config.mask_loss_weight
    return (
        _resolve_stage1_weight("lambda_kl", lambda_kl, config_kl, default=1.0),
        _resolve_stage1_weight("lambda_mask", lambda_mask, config_mask, default=0.0),
    )


def _resolve_stage1_weight(
    name: str,
    explicit_value: float | None,
    config_value: float | None,
    *,
    default: float,
) -> float:
    value = explicit_value
    if value is None:
        value = config_value
    if value is None:
        value = default
    _validate_non_negative_weight(name, value)
    return float(value)


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
    if output.posterior.mu.device != output.logits.device:
        raise ValueError(
            "output.posterior.mu and output.logits must be on the same device"
        )
    if output.posterior.logvar.device != output.logits.device:
        raise ValueError(
            "output.posterior.logvar and output.logits must be on the same device"
        )


def _validate_attention_mask(
    attention_mask: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    require_any: bool,
) -> None:
    if attention_mask.shape != token_ids.shape:
        raise ValueError("attention_mask must match token_ids shape")
    if attention_mask.device != token_ids.device:
        raise ValueError("attention_mask and token_ids must be on the same device")
    if attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be a boolean tensor")
    if require_any and not attention_mask.any().item():
        raise ValueError("attention_mask must select at least one token")


def _validate_non_negative_weight(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _validate_mask_token_id(mask_token_id: int) -> None:
    if not isinstance(mask_token_id, int) or isinstance(mask_token_id, bool):
        raise ValueError("mask_token_id must be an integer")
    if mask_token_id < 0:
        raise ValueError(f"mask_token_id must be non-negative, got {mask_token_id!r}")


def _validate_mask_probability(mask_probability: float) -> None:
    if not isinstance(mask_probability, Real) or isinstance(mask_probability, bool):
        raise ValueError("mask_probability must be a real number")
    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError(
            "mask_probability must be in the range [0, 1], "
            f"got {mask_probability!r}"
        )


def _validate_ignore_index(ignore_index: int) -> None:
    if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
        raise ValueError("ignore_index must be an integer")


def _validate_token_ids_for_masking(token_ids: torch.Tensor) -> None:
    if token_ids.ndim != 2:
        raise ValueError("token_ids must be shaped [batch, seq]")
    if token_ids.dtype != torch.long:
        raise ValueError("token_ids must be a torch.long tensor")
    if token_ids.numel() > 0 and (token_ids < 0).any().item():
        raise ValueError("token_ids must be non-negative")


def _validate_mask_labels(mask_labels: torch.Tensor, token_ids: torch.Tensor) -> None:
    if mask_labels.shape != token_ids.shape:
        raise ValueError("mask_labels must match token_ids shape")
    if mask_labels.device != token_ids.device:
        raise ValueError("mask_labels and token_ids must be on the same device")
    if mask_labels.dtype != torch.long:
        raise ValueError("mask_labels must be a torch.long tensor")


__all__ = (
    "Stage1VAELoss",
    "Stage1MaskingPolicy",
    "apply_stage1_masking",
    "compute_stage1_vae_loss",
    "stage1_pretraining_step",
)
