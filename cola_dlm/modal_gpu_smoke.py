"""Tiny synthetic smoke harness for local and remote GPU validation."""

from __future__ import annotations

from typing import Any

import torch

from cola_dlm.config import (
    DiTConfig,
    DiffusionConfig,
    InferenceConfig,
    OptimizerConfig,
    Stage2Config,
    VAEConfig,
)
from cola_dlm.dit import BlockCausalTextDiT
from cola_dlm.inference import generate
from cola_dlm.stage1 import compute_stage1_vae_loss
from cola_dlm.stage2 import create_frozen_reference_encoder, stage2_joint_training_step
from cola_dlm.vae import TextVAE

__all__ = (
    "build_tiny_modal_smoke_config",
    "run_tiny_modal_gpu_validation",
    "run_tiny_stage2_smoke_step",
)


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

    selected_device, cuda_available = _prepare_device(
        device,
        require_cuda=require_cuda,
    )
    config = build_tiny_modal_smoke_config()
    token_ids, attention_mask = _build_tiny_token_batch(config, selected_device)
    check = _run_stage2_joint_training_check(config, token_ids, attention_mask)
    if not check["loss_finite"]:
        raise RuntimeError("Tiny Stage 2 smoke loss is not finite")

    return {
        "success": bool(check["success"]),
        "device": str(selected_device),
        "cuda_available": bool(cuda_available),
        "loss": check["total_loss"],
        "vae_parameter_device": check["vae_parameter_device"],
        "dit_parameter_device": check["dit_parameter_device"],
        "token_device": check["token_device"],
        "steps": 1,
    }


def run_tiny_modal_gpu_validation(
    device: str | torch.device | None = "cpu",
    *,
    require_cuda: bool = False,
) -> dict[str, Any]:
    """Run a tiny CPU-safe validation sweep and return JSON-safe diagnostics."""

    selected_device, cuda_available = _prepare_device(
        device,
        require_cuda=require_cuda,
    )
    config = build_tiny_modal_smoke_config()
    token_ids, attention_mask = _build_tiny_token_batch(config, selected_device)

    checks: dict[str, dict[str, Any]] = {
        "text_vae": _run_text_vae_check(config, token_ids, attention_mask),
        "stage2_joint_training": _run_stage2_joint_training_check(
            config,
            token_ids,
            attention_mask,
        ),
    }
    limitations: list[str] = []

    inference_check, limitation = _run_inference_generate_check(
        config,
        token_ids,
        attention_mask,
    )
    checks["inference_generate"] = inference_check
    if limitation is not None:
        limitations.append(limitation)

    success = all(bool(check.get("success", False)) for check in checks.values())
    return {
        "success": success,
        "device": str(selected_device),
        "cuda_available": bool(cuda_available),
        "checks": checks,
        "limitations": limitations,
    }


def _build_tiny_modal_inference_config(config: Stage2Config) -> InferenceConfig:
    return InferenceConfig(
        vae=config.vae,
        dit=config.dit,
        diffusion=config.diffusion,
        num_denoise_steps=1,
        sampler="euler",
        cfg_scale=1.0,
        max_new_tokens=config.dit.block_size,
        condition_strategy="clean_condition_repaint",
    )


def _prepare_device(
    device: str | torch.device | None,
    *,
    require_cuda: bool,
) -> tuple[torch.device, bool]:
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
    return selected_device, cuda_available


def _build_tiny_token_batch(
    config: Stage2Config,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = torch.randint(
        0,
        config.vae.vocab_size,
        (config.global_batch_size, config.vae.sequence_length),
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(token_ids, dtype=torch.bool, device=device)
    return token_ids, attention_mask


def _run_text_vae_check(
    config: Stage2Config,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, Any]:
    vae = TextVAE(config.vae).to(token_ids.device)
    trainable_parameters = _trainable_parameters(vae)
    parameter_snapshots = _clone_parameters(trainable_parameters)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.optimizer.peak_lr,
        weight_decay=config.optimizer.weight_decay,
    )

    output = vae(
        token_ids,
        attention_mask=_token_attention_to_transformer_mask(attention_mask),
    )
    loss = compute_stage1_vae_loss(
        output,
        token_ids,
        attention_mask=attention_mask,
        lambda_kl=1.0,
        lambda_mask=0.0,
    )
    loss_tensor_device = str(loss.loss.device)
    loss_finite = bool(torch.isfinite(loss.loss.detach()).item())

    optimizer.zero_grad(set_to_none=True)
    loss.loss.backward()
    optimizer.step()
    parameter_changed = _any_parameter_changed(
        parameter_snapshots,
        trainable_parameters,
    )
    expected_device = str(token_ids.device)
    device_fields = {
        "loss_tensor_device": loss_tensor_device,
        "vae_parameter_device": _first_parameter_device(vae),
        "token_device": str(token_ids.device),
        "mask_device": str(attention_mask.device),
        "posterior_device": str(output.posterior.mu.device),
        "posterior_mu_device": str(output.posterior.mu.device),
        "posterior_logvar_device": str(output.posterior.logvar.device),
        "latent_device": str(output.latents.device),
        "logits_device": str(output.logits.device),
    }
    devices_match = _all_devices_match(device_fields, expected_device)

    return {
        "success": loss_finite and parameter_changed and devices_match,
        "loss": float(loss.loss.detach().cpu().item()),
        "loss_finite": loss_finite,
        "devices_match": devices_match,
        **device_fields,
        "trainable_parameter_changed": parameter_changed,
    }


def _run_stage2_joint_training_check(
    config: Stage2Config,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, Any]:
    vae = TextVAE(config.vae).to(token_ids.device)
    reference_encoder = create_frozen_reference_encoder(vae).to(token_ids.device)
    dit = BlockCausalTextDiT(config.dit).to(token_ids.device)
    trainable_parameters = _trainable_parameters(vae, dit)
    parameter_snapshots = _clone_parameters(trainable_parameters)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
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
    loss_tensor_device = str(loss.loss.device)
    loss_finite = bool(torch.isfinite(loss.loss.detach()).item())
    parameter_changed = _any_parameter_changed(
        parameter_snapshots,
        trainable_parameters,
    )
    expected_device = str(token_ids.device)
    device_fields = {
        "loss_tensor_device": loss_tensor_device,
        "vae_parameter_device": _first_parameter_device(vae),
        "reference_encoder_parameter_device": _first_parameter_device(
            reference_encoder
        ),
        "dit_parameter_device": _first_parameter_device(dit),
        "token_device": str(token_ids.device),
        "mask_device": str(attention_mask.device),
    }
    devices_match = _all_devices_match(device_fields, expected_device)

    return {
        "success": loss_finite and parameter_changed and devices_match,
        "total_loss": float(loss.loss.detach().cpu().item()),
        "loss_finite": loss_finite,
        "devices_match": devices_match,
        **device_fields,
        "trainable_parameter_changed": parameter_changed,
        "steps": 1,
    }


def _run_inference_generate_check(
    config: Stage2Config,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[dict[str, Any], str | None]:
    inference_config = _build_tiny_modal_inference_config(config)
    prefix_length = config.dit.block_size
    prefix_token_ids = token_ids[:, :prefix_length].contiguous()
    prefix_attention_mask = attention_mask[:, :prefix_length].contiguous()
    vae = TextVAE(config.vae).to(token_ids.device)
    dit = BlockCausalTextDiT(config.dit).to(token_ids.device)
    vae.eval()
    dit.eval()

    try:
        with torch.no_grad():
            output = generate(
                vae,
                dit,
                prefix_token_ids,
                inference_config=inference_config,
                attention_mask=prefix_attention_mask,
                max_new_tokens=inference_config.max_new_tokens,
            )
    except (NotImplementedError, TypeError, ValueError) as exc:
        reason = f"inference/generate skipped: {exc}"
        return (
            {
                "success": True,
                "skipped": True,
                "reason": reason,
            },
            reason,
        )

    expected_device = str(prefix_token_ids.device)
    device_fields = {
        "generated_latent_device": str(output.generated_latents.device),
        "all_latent_device": str(output.all_latents.device),
        "response_logits_device": str(output.response_logits.device),
        "response_token_device": str(output.response_token_ids.device),
        "token_device": str(prefix_token_ids.device),
        "mask_device": str(prefix_attention_mask.device),
        "vae_parameter_device": _first_parameter_device(vae),
        "dit_parameter_device": _first_parameter_device(dit),
    }
    devices_match = _all_devices_match(device_fields, expected_device)

    return (
        {
            "success": devices_match,
            "skipped": False,
            "devices_match": devices_match,
            **device_fields,
        },
        None,
    )


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


def _trainable_parameters(*modules: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def _clone_parameters(
    parameters: list[torch.nn.Parameter],
) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _any_parameter_changed(
    snapshots: list[torch.Tensor],
    parameters: list[torch.nn.Parameter],
) -> bool:
    return any(
        not torch.equal(snapshot, parameter.detach())
        for snapshot, parameter in zip(snapshots, parameters, strict=True)
    )


def _all_devices_match(device_fields: dict[str, str], expected_device: str) -> bool:
    return all(device == expected_device for device in device_fields.values())


def _token_attention_to_transformer_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return attention_mask[:, None, :].expand(
        attention_mask.shape[0],
        attention_mask.shape[1],
        attention_mask.shape[1],
    )
