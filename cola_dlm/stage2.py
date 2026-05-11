"""Stage 2 joint VAE-DiT training helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Real

import torch
import torch.nn.functional as F

from cola_dlm.config import Stage2Config
from cola_dlm.dit import BlockCausalTextDiT
from cola_dlm.vae import (
    DiagonalGaussianPosterior,
    TextVAE,
    TextVAEEncoder,
    TextVAEOutput,
    vae_logsnr,
)


_DEFAULT_MASK_IGNORE_INDEX = -100


@dataclass(frozen=True)
class Stage2Loss:
    """Structured scalar losses and diagnostics for Stage 2 training."""

    loss: torch.Tensor
    vae_loss: torch.Tensor
    flow_matching_loss: torch.Tensor
    reference_kl: torch.Tensor
    reconstruction_nll: torch.Tensor
    posterior_regularizer: torch.Tensor
    mask_loss: torch.Tensor
    logsnr: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return diagnostics with stable public names."""

        return {
            "loss": self.loss,
            "vae_loss": self.vae_loss,
            "flow_matching_loss": self.flow_matching_loss,
            "reference_kl": self.reference_kl,
            "reconstruction_nll": self.reconstruction_nll,
            "posterior_regularizer": self.posterior_regularizer,
            "mask_loss": self.mask_loss,
            "logsnr": self.logsnr,
        }


def create_frozen_reference_encoder(vae: TextVAE) -> TextVAEEncoder:
    """Return an eval-mode, gradient-frozen copy of ``vae.encoder``."""

    if not isinstance(vae, TextVAE):
        raise TypeError("vae must be a TextVAE")

    reference_encoder = copy.deepcopy(vae.encoder)
    reference_encoder.eval()
    for parameter in reference_encoder.parameters():
        parameter.requires_grad_(False)
    return reference_encoder


def reference_kl(
    trainable_posterior: DiagonalGaussianPosterior,
    reference_posterior: DiagonalGaussianPosterior,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return mean KL(q_trainable || q_reference) for diagonal Gaussians."""

    _validate_posterior_pair(trainable_posterior, reference_posterior)
    if attention_mask is not None:
        _validate_attention_mask_for_shape(
            attention_mask,
            trainable_posterior.mu.shape[:-1],
        )
        if attention_mask.device != trainable_posterior.mu.device:
            raise ValueError(
                "attention_mask and posterior tensors must be on the same device"
            )

    trainable_var = torch.exp(trainable_posterior.logvar)
    reference_inv_var = torch.exp(-reference_posterior.logvar)
    mean_delta = trainable_posterior.mu - reference_posterior.mu
    per_dim_kl = (
        reference_posterior.logvar
        - trainable_posterior.logvar
        + (trainable_var + mean_delta.pow(2)) * reference_inv_var
        - 1.0
    )
    per_token_kl = 0.5 * per_dim_kl.sum(dim=-1)
    return _masked_mean(per_token_kl, attention_mask)


def compute_stage2_vae_loss(
    output: TextVAEOutput,
    token_ids: torch.Tensor,
    reference_posterior: DiagonalGaussianPosterior,
    *,
    attention_mask: torch.Tensor | None = None,
    mask_labels: torch.Tensor | None = None,
    stage2_config: Stage2Config | None = None,
    lambda_vae: float | None = None,
    lambda_flow_matching: float | None = None,
    lambda_reference_kl: float | None = None,
    lambda_posterior_regularizer: float | None = None,
    lambda_mask: float | None = None,
    mask_ignore_index: int = _DEFAULT_MASK_IGNORE_INDEX,
) -> Stage2Loss:
    """Compute Stage 2 VAE-side losses before the DiT prior is wired."""

    (
        lambda_vae,
        lambda_flow_matching,
        lambda_reference_kl,
        lambda_posterior_regularizer,
        lambda_mask,
    ) = _resolve_stage2_weights(
        stage2_config=stage2_config,
        lambda_vae=lambda_vae,
        lambda_flow_matching=lambda_flow_matching,
        lambda_reference_kl=lambda_reference_kl,
        lambda_posterior_regularizer=lambda_posterior_regularizer,
        lambda_mask=lambda_mask,
    )
    _validate_ignore_index(mask_ignore_index)
    _validate_stage2_vae_loss_shapes(
        output=output,
        token_ids=token_ids,
        reference_posterior=reference_posterior,
    )
    if attention_mask is not None:
        _validate_attention_mask(attention_mask, token_ids, require_any=True)
    if mask_labels is not None:
        _validate_mask_labels(mask_labels, token_ids)

    reconstruction_nll = _masked_mean(
        _per_token_cross_entropy(output.logits, token_ids),
        attention_mask,
    )
    posterior_regularizer = _masked_mean(
        -output.posterior.log_prob(output.latents),
        attention_mask,
    )
    mask_loss = _zero_scalar_like(output.logits)
    if mask_labels is not None:
        mask_loss = _masked_cross_entropy(
            output.logits,
            mask_labels,
            ignore_index=mask_ignore_index,
            attention_mask=attention_mask,
        )
    reference_kl_loss = reference_kl(
        output.posterior,
        reference_posterior,
        attention_mask=attention_mask,
    )
    logsnr = vae_logsnr(output.posterior.mu, output.posterior.logvar)
    flow_matching_loss = _zero_scalar_like(output.logits)
    vae_loss = (
        reconstruction_nll
        + lambda_posterior_regularizer * posterior_regularizer
        + lambda_mask * mask_loss
    )
    loss = (
        lambda_vae * vae_loss
        + lambda_flow_matching * flow_matching_loss
        + lambda_reference_kl * reference_kl_loss
    )
    return Stage2Loss(
        loss=loss,
        vae_loss=vae_loss,
        flow_matching_loss=flow_matching_loss,
        reference_kl=reference_kl_loss,
        reconstruction_nll=reconstruction_nll,
        posterior_regularizer=posterior_regularizer,
        mask_loss=mask_loss,
        logsnr=logsnr,
    )


def _validate_stage2_component_shapes(
    *,
    vae: TextVAE,
    reference_encoder: TextVAEEncoder,
    dit: BlockCausalTextDiT,
    stage2_config: Stage2Config,
) -> None:
    """Check module token-latent boundaries against the Stage 2 config."""

    _require_instance("vae", vae, TextVAE)
    _require_instance("reference_encoder", reference_encoder, TextVAEEncoder)
    _require_instance("dit", dit, BlockCausalTextDiT)
    _require_instance("stage2_config", stage2_config, Stage2Config)

    _require_equal(
        "vae.encoder.latent_dim",
        vae.encoder.latent_dim,
        "vae.decoder.latent_dim",
        vae.decoder.latent_dim,
    )
    _require_equal(
        "vae.encoder.latent_dim",
        vae.encoder.latent_dim,
        "reference_encoder.latent_dim",
        reference_encoder.latent_dim,
    )
    _require_equal(
        "vae.encoder.latent_dim",
        vae.encoder.latent_dim,
        "stage2_config.vae.latent_dim",
        stage2_config.vae.latent_dim,
    )
    _require_equal(
        "dit.config.latent_dim",
        dit.config.latent_dim,
        "stage2_config.dit.latent_dim",
        stage2_config.dit.latent_dim,
    )
    _require_equal(
        "stage2_config.vae.latent_dim",
        stage2_config.vae.latent_dim,
        "stage2_config.dit.latent_dim",
        stage2_config.dit.latent_dim,
    )
    _require_equal(
        "stage2_config.vae.sequence_length",
        stage2_config.vae.sequence_length,
        "stage2_config.dit.sequence_length",
        stage2_config.dit.sequence_length,
    )
    _require_equal(
        "dit.config.sequence_length",
        dit.config.sequence_length,
        "stage2_config.dit.sequence_length",
        stage2_config.dit.sequence_length,
    )
    _require_equal(
        "dit.config.block_size",
        dit.config.block_size,
        "stage2_config.dit.block_size",
        stage2_config.dit.block_size,
    )
    _require_equal(
        "vae.encoder.vocab_size",
        vae.encoder.vocab_size,
        "vae.decoder.vocab_size",
        vae.decoder.vocab_size,
    )
    _require_equal(
        "vae.encoder.vocab_size",
        vae.encoder.vocab_size,
        "reference_encoder.vocab_size",
        reference_encoder.vocab_size,
    )
    _require_equal(
        "vae.encoder.vocab_size",
        vae.encoder.vocab_size,
        "stage2_config.vae.vocab_size",
        stage2_config.vae.vocab_size,
    )
    _require_equal(
        "vae.encoder.patch_size",
        vae.encoder.patch_size,
        "vae.decoder.patch_size",
        vae.decoder.patch_size,
    )
    _require_equal(
        "vae.encoder.patch_size",
        vae.encoder.patch_size,
        "reference_encoder.patch_size",
        reference_encoder.patch_size,
    )
    _require_equal(
        "vae.encoder.patch_size",
        vae.encoder.patch_size,
        "stage2_config.vae.patch_size",
        stage2_config.vae.patch_size,
    )


def _require_instance(name: str, value: object, expected_type: type[object]) -> None:
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be a {expected_type.__name__}")


def _require_equal(
    left_name: str,
    left_value: int,
    right_name: str,
    right_value: int,
) -> None:
    if left_value != right_value:
        raise ValueError(
            f"{left_name} must match {right_name} "
            f"(got {left_value!r} and {right_value!r})"
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


def _resolve_stage2_weights(
    *,
    stage2_config: Stage2Config | None,
    lambda_vae: float | None,
    lambda_flow_matching: float | None,
    lambda_reference_kl: float | None,
    lambda_posterior_regularizer: float | None,
    lambda_mask: float | None,
) -> tuple[float, float, float, float, float]:
    config_vae = None if stage2_config is None else stage2_config.vae_loss_weight
    config_flow = (
        None if stage2_config is None else stage2_config.flow_matching_loss_weight
    )
    config_reference_kl = (
        None if stage2_config is None else stage2_config.reference_kl_weight
    )
    return (
        _resolve_stage2_weight("lambda_vae", lambda_vae, config_vae, default=1.0),
        _resolve_stage2_weight(
            "lambda_flow_matching",
            lambda_flow_matching,
            config_flow,
            default=1.0,
        ),
        _resolve_stage2_weight(
            "lambda_reference_kl",
            lambda_reference_kl,
            config_reference_kl,
            default=1.0,
        ),
        _resolve_stage2_weight(
            "lambda_posterior_regularizer",
            lambda_posterior_regularizer,
            None,
            default=1.0,
        ),
        _resolve_stage2_weight("lambda_mask", lambda_mask, None, default=0.0),
    )


def _resolve_stage2_weight(
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


def _validate_stage2_vae_loss_shapes(
    *,
    output: TextVAEOutput,
    token_ids: torch.Tensor,
    reference_posterior: DiagonalGaussianPosterior,
) -> None:
    if output.logits.ndim != 3:
        raise ValueError("output.logits must be shaped [batch, seq, vocab]")
    if token_ids.ndim != 2:
        raise ValueError("token_ids must be shaped [batch, seq]")
    if token_ids.dtype != torch.long:
        raise ValueError("token_ids must be a torch.long tensor")
    if output.logits.shape[:2] != token_ids.shape:
        raise ValueError("output.logits and token_ids must share batch and seq shape")
    if output.logits.device != token_ids.device:
        raise ValueError("output.logits and token_ids must be on the same device")
    if output.latents.shape != output.posterior.mu.shape:
        raise ValueError("output.latents must match output.posterior.mu shape")
    if output.posterior.mu.ndim != 3:
        raise ValueError("output.posterior tensors must be shaped [batch, seq, latent]")
    if output.posterior.mu.shape[:2] != token_ids.shape:
        raise ValueError(
            "output.posterior and token_ids must share batch and seq shape"
        )
    if output.latents.device != output.logits.device:
        raise ValueError("output.latents and output.logits must be on the same device")
    if output.posterior.mu.device != output.logits.device:
        raise ValueError(
            "output.posterior.mu and output.logits must be on the same device"
        )
    if output.posterior.logvar.device != output.logits.device:
        raise ValueError(
            "output.posterior.logvar and output.logits must be on the same device"
        )
    _validate_posterior_pair(output.posterior, reference_posterior)


def _validate_posterior_pair(
    trainable_posterior: DiagonalGaussianPosterior,
    reference_posterior: DiagonalGaussianPosterior,
) -> None:
    _require_instance(
        "trainable_posterior",
        trainable_posterior,
        DiagonalGaussianPosterior,
    )
    _require_instance(
        "reference_posterior",
        reference_posterior,
        DiagonalGaussianPosterior,
    )
    if trainable_posterior.mu.shape != reference_posterior.mu.shape:
        raise ValueError("posterior tensors must share shape")
    if trainable_posterior.mu.device != reference_posterior.mu.device:
        raise ValueError("posterior tensors must be on the same device")


def _validate_attention_mask(
    attention_mask: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    require_any: bool,
) -> None:
    _validate_attention_mask_for_shape(attention_mask, token_ids.shape)
    if attention_mask.device != token_ids.device:
        raise ValueError("attention_mask and token_ids must be on the same device")
    if require_any and not attention_mask.any().item():
        raise ValueError("attention_mask must select at least one token")


def _validate_attention_mask_for_shape(
    attention_mask: torch.Tensor,
    expected_shape: torch.Size | tuple[int, ...],
) -> None:
    if attention_mask.shape != expected_shape:
        raise ValueError("attention_mask must match token batch shape")
    if attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be a boolean tensor")
    if not attention_mask.any().item():
        raise ValueError("attention_mask must select at least one token")


def _validate_mask_labels(mask_labels: torch.Tensor, token_ids: torch.Tensor) -> None:
    if mask_labels.shape != token_ids.shape:
        raise ValueError("mask_labels must match token_ids shape")
    if mask_labels.device != token_ids.device:
        raise ValueError("mask_labels and token_ids must be on the same device")
    if mask_labels.dtype != torch.long:
        raise ValueError("mask_labels must be a torch.long tensor")


def _validate_non_negative_weight(name: str, value: float) -> None:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real number")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _validate_ignore_index(ignore_index: int) -> None:
    if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
        raise ValueError("ignore_index must be an integer")


__all__ = (
    "Stage2Loss",
    "create_frozen_reference_encoder",
    "reference_kl",
    "compute_stage2_vae_loss",
)
