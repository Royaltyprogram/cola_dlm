"""Tiny synthetic Stage 2 smoke harness for local and remote GPU checks."""

from __future__ import annotations

from typing import Any

import torch

from cola_dlm.config import (
    DiTConfig,
    DiffusionConfig,
    OptimizerConfig,
    Stage2Config,
    VAEConfig,
)
from cola_dlm.dit import BlockCausalTextDiT
from cola_dlm.stage2 import create_frozen_reference_encoder, stage2_joint_training_step
from cola_dlm.vae import TextVAE

__all__ = ("build_tiny_modal_smoke_config", "run_tiny_stage2_smoke_step")


def build_tiny_modal_smoke_config() -> Stage2Config:
    """Return the smallest Stage 2 config used by the Modal smoke harness."""

    vae_config = VAEConfig(
        tokenizer_name="modal-smoke",
        vocab_size=32,
        sequence_length=8,
        latent_dim=2,
        patch_size=1,
        encoder_layers=1,
        decoder_layers=1,
        hidden_size=8,
        ffn_size=16,
        num_attention_heads=1,
        attention_head_dim=8,
    )
    dit_config = DiTConfig(
        sequence_length=8,
        latent_dim=2,
        block_size=2,
        num_layers=1,
        hidden_size=8,
        ffn_size=16,
        num_attention_heads=1,
        attention_head_dim=8,
    )
    return Stage2Config(
        vae=vae_config,
        dit=dit_config,
        diffusion=DiffusionConfig(logit_normal_loc=0.0, logit_normal_scale=0.5),
        optimizer=OptimizerConfig(peak_lr=1.0e-3, weight_decay=0.0, warmup_steps=0),
        global_batch_size=2,
        tokens_per_step=16,
        vae_loss_weight=1.0,
        flow_matching_loss_weight=1.0,
        reference_kl_weight=1.0,
    )


def run_tiny_stage2_smoke_step(
    device: str | torch.device | None = "cpu",
    *,
    require_cuda: bool = False,
) -> dict[str, Any]:
    """Run one synthetic Stage 2 optimizer step and return JSON-safe diagnostics."""

    selected_device = torch.device("cpu" if device is None else device)
    cuda_available = torch.cuda.is_available()
    _validate_device(
        selected_device,
        require_cuda=require_cuda,
        cuda_available=cuda_available,
    )

    torch.manual_seed(0)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    config = build_tiny_modal_smoke_config()
    vae = TextVAE(config.vae).to(selected_device)
    reference_encoder = create_frozen_reference_encoder(vae).to(selected_device)
    dit = BlockCausalTextDiT(config.dit).to(selected_device)

    token_ids = torch.randint(
        0,
        config.vae.vocab_size,
        (config.global_batch_size, config.vae.sequence_length),
        device=selected_device,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(token_ids, dtype=torch.bool, device=selected_device)
    optimizer = torch.optim.AdamW(
        list(vae.parameters()) + list(dit.parameters()),
        lr=config.optimizer.peak_lr,
        weight_decay=config.optimizer.weight_decay,
    )

    loss = stage2_joint_training_step(
        vae,
        reference_encoder,
        dit,
        optimizer,
        token_ids,
        attention_mask=attention_mask,
        stage2_config=config,
        max_grad_norm=config.optimizer.grad_clip,
    )
    loss_value = float(loss.loss.detach().cpu().item())
    if not torch.isfinite(loss.loss.detach()).item():
        raise RuntimeError("Tiny Stage 2 smoke loss is not finite")

    return {
        "success": True,
        "device": str(selected_device),
        "cuda_available": bool(cuda_available),
        "loss": loss_value,
        "vae_parameter_device": _first_parameter_device(vae),
        "dit_parameter_device": _first_parameter_device(dit),
        "token_device": str(token_ids.device),
        "steps": 1,
    }


def _validate_device(
    device: torch.device,
    *,
    require_cuda: bool,
    cuda_available: bool,
) -> None:
    if require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA is required for the tiny Stage 2 smoke step, but CUDA is unavailable"
        )
    if require_cuda and device.type != "cuda":
        raise RuntimeError(
            "CUDA is required for the tiny Stage 2 smoke step, "
            f"but selected device is {device!s}"
        )
    if device.type == "cuda" and not cuda_available:
        raise RuntimeError(
            f"Selected device {device!s} requires CUDA, but CUDA is unavailable"
        )


def _first_parameter_device(module: torch.nn.Module) -> str:
    return str(next(module.parameters()).device)
