"""Reusable diagnostics for VAE and latent health checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cola_dlm.block_causal_mask import NOISY_SEGMENT_ID
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


def compute_flow_matching_loss_by_block(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    block_ids: torch.Tensor,
    segment_ids: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Return detached Flow Matching MSE diagnostics grouped by noisy block id."""

    _validate_flow_matching_block_inputs(
        prediction,
        target,
        loss_mask,
        block_ids,
        segment_ids,
    )

    noisy_positions = segment_ids == NOISY_SEGMENT_ID
    selected_positions = loss_mask & noisy_positions.unsqueeze(0)
    if not selected_positions.any().item():
        raise ValueError("loss_mask must select at least one noisy target position")

    selected_block_ids = block_ids[selected_positions.any(dim=0)]
    block_losses: dict[int, torch.Tensor] = {}
    squared_error = (prediction.detach() - target.detach()).square()
    for block_id in torch.sort(torch.unique(selected_block_ids)).values:
        block_index = int(block_id.item())
        block_mask = selected_positions & (block_ids == block_index).unsqueeze(0)
        block_losses[block_index] = squared_error[block_mask].mean().detach()
    return block_losses


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


def _validate_flow_matching_block_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    block_ids: torch.Tensor,
    segment_ids: torch.Tensor,
) -> None:
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("prediction must be a torch.Tensor")
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a torch.Tensor")
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have matching shapes, "
            f"got {prediction.shape} and {target.shape}"
        )
    if prediction.ndim != 3:
        raise ValueError(
            "prediction and target must be shaped [batch, packed_len, latent]"
        )
    if not prediction.is_floating_point():
        raise TypeError("prediction must be a floating point tensor")
    if not target.is_floating_point():
        raise TypeError("target must be a floating point tensor")
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")

    packed_shape = prediction.shape[:2]
    if not isinstance(loss_mask, torch.Tensor):
        raise TypeError("loss_mask must be a torch.Tensor")
    if loss_mask.shape != packed_shape:
        raise ValueError("loss_mask must be shaped [batch, packed_len]")
    if loss_mask.dtype != torch.bool:
        raise ValueError("loss_mask must be a boolean tensor")
    if loss_mask.device != prediction.device:
        raise ValueError("loss_mask and prediction must be on the same device")

    packed_len = prediction.shape[1]
    _validate_packed_index_tensor(block_ids, "block_ids", packed_len, prediction.device)
    _validate_packed_index_tensor(
        segment_ids,
        "segment_ids",
        packed_len,
        prediction.device,
    )
    if torch.any(block_ids < 0).item():
        raise ValueError("block_ids must be non-negative")


def _validate_packed_index_tensor(
    tensor: torch.Tensor,
    name: str,
    packed_len: int,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.shape != (packed_len,):
        raise ValueError(f"{name} must be shaped [packed_len]")
    if tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} must be an integer tensor")
    if tensor.device != device:
        raise ValueError(f"{name} and prediction must be on the same device")


__all__ = (
    "VAEDiagnostics",
    "compute_flow_matching_loss_by_block",
    "compute_vae_diagnostics",
)
