"""Public inference helpers for block-wise Cola DLM generation."""

from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch

from cola_dlm.block_causal_mask import (
    CLEAN_SEGMENT_ID,
    NOISY_SEGMENT_ID,
    build_block_causal_attention_mask,
)
from cola_dlm.config import InferenceConfig
from cola_dlm.dit import BlockCausalTextDiT
from cola_dlm.vae import TextVAE


def _reject_kv_cache(kv_cache: object | None) -> None:
    if kv_cache is not None:
        raise ValueError("kv_cache is not supported yet; pass None")


@dataclass(frozen=True)
class InferenceOutput:
    """Container returned by the inference path."""

    prefix_latents: Any
    generated_latents: Any
    all_latents: Any
    response_logits: Any
    response_token_ids: Any
    kv_cache: None = None

    def __post_init__(self) -> None:
        _reject_kv_cache(self.kv_cache)


@dataclass(frozen=True)
class GenerationBlock:
    """Absolute block range and the prefix positions known inside that range."""

    start: int
    end: int
    known_mask: torch.Tensor

    def __iter__(self) -> Iterator[Any]:
        yield self.start
        yield self.end
        yield self.known_mask


@dataclass(frozen=True)
class _PackedInferenceBlock:
    latents: torch.Tensor
    attention_mask: torch.Tensor
    block_ids: torch.Tensor
    segment_ids: torch.Tensor
    current_slice: slice


def encode_prefix_latents(
    vae: TextVAE,
    prefix_token_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    deterministic: bool = True,
) -> torch.Tensor:
    """Encode prefix tokens into clean per-token VAE latents."""

    _validate_prefix_token_ids(prefix_token_ids)
    if attention_mask is not None:
        _validate_prefix_attention_mask(attention_mask, prefix_token_ids)

    posterior = vae.encoder(
        prefix_token_ids,
        attention_mask=_token_attention_to_transformer_mask(attention_mask),
    )
    if deterministic:
        return posterior.mode()
    return posterior.sample(generator=generator)


def generate(
    vae: TextVAE,
    dit: BlockCausalTextDiT,
    prefix_token_ids: torch.Tensor,
    *,
    inference_config: InferenceConfig | None = None,
    max_new_tokens: int | None = None,
    attention_mask: torch.Tensor | None = None,
    unconditional_prefix_token_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    decoder_fill_token_id: int = 0,
    kv_cache: None = None,
) -> InferenceOutput:
    """Generate response latents and decode greedy response token ids."""

    _reject_kv_cache(kv_cache)
    config = inference_config if inference_config is not None else InferenceConfig()
    max_new_tokens = _resolve_positive_int(
        "max_new_tokens",
        max_new_tokens,
        config.max_new_tokens,
    )
    cfg_scale = _resolve_non_negative_float("cfg_scale", None, config.cfg_scale)
    _validate_clean_condition_strategy(config.condition_strategy)
    _validate_prefix_token_ids(prefix_token_ids)
    if attention_mask is not None:
        _validate_prefix_attention_mask(attention_mask, prefix_token_ids)
    _validate_generation_fits_dit(
        prefix_length=prefix_token_ids.shape[1],
        num_new_latents=max_new_tokens,
        sequence_length=config.dit.sequence_length,
    )

    prefix_latents = encode_prefix_latents(
        vae,
        prefix_token_ids,
        attention_mask=attention_mask,
        generator=generator,
    )
    unconditional_prefix_latents = None
    if unconditional_prefix_token_ids is not None and cfg_scale != 1.0:
        _validate_unconditional_prefix_token_ids(
            unconditional_prefix_token_ids,
            prefix_token_ids,
        )
        unconditional_prefix_latents = encode_prefix_latents(
            vae,
            unconditional_prefix_token_ids,
            attention_mask=attention_mask,
            generator=generator,
        )

    generated_latents = sample_latent_blocks(
        dit,
        prefix_latents,
        inference_config=config,
        num_new_latents=max_new_tokens,
        unconditional_prefix_latents=unconditional_prefix_latents,
        generator=generator,
        cfg_scale=cfg_scale,
    )
    all_latents = torch.cat((prefix_latents, generated_latents), dim=1)
    decoder_token_ids = _build_decoder_token_ids(
        prefix_token_ids,
        num_generated_tokens=max_new_tokens,
        fill_token_id=decoder_fill_token_id,
        vocab_size=vae.decoder.vocab_size,
    )
    decoder_attention_mask = _build_decoder_attention_mask(
        attention_mask,
        num_generated_tokens=max_new_tokens,
    )
    logits = vae.decoder(
        decoder_token_ids,
        all_latents,
        attention_mask=decoder_attention_mask,
    )
    response_logits = logits[:, prefix_token_ids.shape[1] :, :]
    response_token_ids = response_logits.argmax(dim=-1)

    return InferenceOutput(
        prefix_latents=prefix_latents,
        generated_latents=generated_latents,
        all_latents=all_latents,
        response_logits=response_logits,
        response_token_ids=response_token_ids,
        kv_cache=None,
    )


def sample_latent_blocks(
    dit: BlockCausalTextDiT,
    prefix_latents: torch.Tensor,
    *,
    inference_config: InferenceConfig | None = None,
    num_new_latents: int | None = None,
    unconditional_prefix_latents: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    cfg_scale: float | None = None,
    block_size: int | None = None,
) -> torch.Tensor:
    """Generate response latents block by block with Euler or Heun denoising."""

    config = inference_config if inference_config is not None else InferenceConfig()
    _validate_sampler_inputs(
        dit=dit,
        prefix_latents=prefix_latents,
        unconditional_prefix_latents=unconditional_prefix_latents,
    )
    num_new_latents = _resolve_positive_int(
        "num_new_latents",
        num_new_latents,
        config.max_new_tokens,
    )
    block_size = _resolve_positive_int("block_size", block_size, config.dit.block_size)
    cfg_scale = _resolve_non_negative_float("cfg_scale", cfg_scale, config.cfg_scale)
    _validate_clean_condition_strategy(config.condition_strategy)
    _validate_generation_fits_dit(
        prefix_length=prefix_latents.shape[1],
        num_new_latents=num_new_latents,
        sequence_length=dit.config.sequence_length,
    )

    generated_blocks: list[torch.Tensor] = []
    for block in iter_generation_blocks(
        prefix_length=prefix_latents.shape[1],
        num_new_latents=num_new_latents,
        block_size=block_size,
        condition_strategy=config.condition_strategy,
    ):
        all_clean_latents = _cat_latents(prefix_latents, generated_blocks)
        unconditional_clean_latents = None
        if unconditional_prefix_latents is not None:
            unconditional_clean_latents = _cat_latents(
                unconditional_prefix_latents,
                generated_blocks,
            )

        block_known_mask = block.known_mask.to(device=prefix_latents.device)
        block_latents = _initial_block_latents(
            prefix_latents,
            block,
            block_known_mask,
            generator=generator,
        )
        clean_block_latents = _known_prefix_block_latents(
            prefix_latents,
            block,
            block_latents,
        )
        unconditional_clean_block_latents = None
        if unconditional_prefix_latents is not None:
            unconditional_clean_block_latents = _known_prefix_block_latents(
                unconditional_prefix_latents,
                block,
                block_latents,
            )

        denoised_block = _denoise_inference_block(
            dit=dit,
            clean_history_latents=all_clean_latents[:, : block.start],
            block_latents=block_latents,
            block_start=block.start,
            known_mask=block_known_mask,
            clean_block_latents=clean_block_latents,
            config=config,
            cfg_scale=cfg_scale,
            block_size=block_size,
            unconditional_clean_history_latents=(
                None
                if unconditional_clean_latents is None
                else unconditional_clean_latents[:, : block.start]
            ),
            unconditional_clean_block_latents=unconditional_clean_block_latents,
        )
        generated_blocks.append(denoised_block[:, ~block_known_mask, :])

    return torch.cat(generated_blocks, dim=1)


def combine_cfg_vector_fields(
    unconditional_field: torch.Tensor,
    conditional_field: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    """Combine conditional and unconditional vector fields with CFG."""

    _validate_matching_vector_fields(unconditional_field, conditional_field)
    cfg_scale = _validate_non_negative_real("cfg_scale", cfg_scale)
    return unconditional_field + cfg_scale * (conditional_field - unconditional_field)


def iter_generation_blocks(
    prefix_length: int,
    num_new_latents: int,
    block_size: int,
    *,
    condition_strategy: str = "clean_condition_repaint",
) -> Iterator[GenerationBlock]:
    """Yield generation block ranges with known prefix positions marked."""

    _validate_clean_condition_strategy(condition_strategy)
    _validate_non_negative_int("prefix_length", prefix_length)
    _validate_positive_int("num_new_latents", num_new_latents)
    _validate_positive_int("block_size", block_size)

    total_length = prefix_length + num_new_latents
    block_start = (prefix_length // block_size) * block_size

    while block_start < total_length:
        block_end = min(block_start + block_size, total_length)
        block_length = block_end - block_start
        known_count = max(0, min(prefix_length, block_end) - block_start)
        known_mask = torch.zeros(block_length, dtype=torch.bool)
        if known_count:
            known_mask[:known_count] = True

        yield GenerationBlock(
            start=block_start,
            end=block_end,
            known_mask=known_mask,
        )
        block_start += block_size


def apply_clean_condition_repaint(
    block_latents: torch.Tensor,
    clean_block_latents: torch.Tensor,
    known_mask: torch.Tensor,
) -> torch.Tensor:
    """Return block latents with known positions copied from clean latents."""

    _validate_repaint_latents(block_latents, clean_block_latents)
    expanded_mask = _expand_known_mask(known_mask, block_latents)
    return torch.where(expanded_mask, clean_block_latents, block_latents)


def _denoise_inference_block(
    *,
    dit: BlockCausalTextDiT,
    clean_history_latents: torch.Tensor,
    block_latents: torch.Tensor,
    block_start: int,
    known_mask: torch.Tensor,
    clean_block_latents: torch.Tensor,
    config: InferenceConfig,
    cfg_scale: float,
    block_size: int,
    unconditional_clean_history_latents: torch.Tensor | None = None,
    unconditional_clean_block_latents: torch.Tensor | None = None,
) -> torch.Tensor:
    block_latents = apply_clean_condition_repaint(
        block_latents,
        clean_block_latents,
        known_mask,
    )
    timesteps = _denoising_time_grid(
        config,
        device=block_latents.device,
        dtype=block_latents.dtype,
    )

    for step_index in range(config.num_denoise_steps):
        timestep = timesteps[step_index]
        next_timestep = timesteps[step_index + 1]
        dt = next_timestep - timestep

        field = _predict_guided_vector_field(
            dit=dit,
            clean_history_latents=clean_history_latents,
            block_latents=block_latents,
            block_start=block_start,
            block_size=block_size,
            timestep=timestep,
            prediction_type=config.diffusion.prediction_type,
            cfg_scale=cfg_scale,
            known_mask=known_mask,
            unconditional_clean_history_latents=unconditional_clean_history_latents,
            unconditional_clean_block_latents=unconditional_clean_block_latents,
        )
        proposal = apply_clean_condition_repaint(
            block_latents + dt * field,
            clean_block_latents,
            known_mask,
        )

        if config.sampler == "heun":
            proposal_field = _predict_guided_vector_field(
                dit=dit,
                clean_history_latents=clean_history_latents,
                block_latents=proposal,
                block_start=block_start,
                block_size=block_size,
                timestep=next_timestep,
                prediction_type=config.diffusion.prediction_type,
                cfg_scale=cfg_scale,
                known_mask=known_mask,
                unconditional_clean_history_latents=(
                    unconditional_clean_history_latents
                ),
                unconditional_clean_block_latents=(
                    unconditional_clean_block_latents
                ),
            )
            block_latents = block_latents + 0.5 * dt * (field + proposal_field)
        else:
            block_latents = proposal

        block_latents = apply_clean_condition_repaint(
            block_latents,
            clean_block_latents,
            known_mask,
        )

    return block_latents


def _predict_guided_vector_field(
    *,
    dit: BlockCausalTextDiT,
    clean_history_latents: torch.Tensor,
    block_latents: torch.Tensor,
    block_start: int,
    block_size: int,
    timestep: torch.Tensor,
    prediction_type: str,
    cfg_scale: float,
    known_mask: torch.Tensor,
    unconditional_clean_history_latents: torch.Tensor | None = None,
    unconditional_clean_block_latents: torch.Tensor | None = None,
) -> torch.Tensor:
    conditional_field = _predict_block_vector_field(
        dit=dit,
        clean_history_latents=clean_history_latents,
        block_latents=block_latents,
        block_start=block_start,
        block_size=block_size,
        timestep=timestep,
        prediction_type=prediction_type,
    )
    if unconditional_clean_history_latents is None or cfg_scale == 1.0:
        return conditional_field

    unconditional_block_latents = block_latents
    if unconditional_clean_block_latents is not None:
        unconditional_block_latents = apply_clean_condition_repaint(
            block_latents,
            unconditional_clean_block_latents,
            known_mask,
        )
    unconditional_field = _predict_block_vector_field(
        dit=dit,
        clean_history_latents=unconditional_clean_history_latents,
        block_latents=unconditional_block_latents,
        block_start=block_start,
        block_size=block_size,
        timestep=timestep,
        prediction_type=prediction_type,
    )
    return combine_cfg_vector_fields(
        unconditional_field,
        conditional_field,
        cfg_scale,
    )


def _predict_block_vector_field(
    *,
    dit: BlockCausalTextDiT,
    clean_history_latents: torch.Tensor,
    block_latents: torch.Tensor,
    block_start: int,
    block_size: int,
    timestep: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    packed = _pack_inference_block(
        clean_history_latents,
        block_latents,
        block_start=block_start,
        block_size=block_size,
    )
    timestep_batch = _expand_timestep(
        timestep,
        batch_size=block_latents.shape[0],
        dtype=block_latents.dtype,
    )
    prediction = dit(
        packed.latents,
        timestep_batch,
        packed.attention_mask,
        packed.segment_ids,
    )
    block_prediction = prediction[:, packed.current_slice, :]
    return _prediction_to_vector_field(
        prediction=block_prediction,
        current_latents=block_latents,
        timestep=timestep_batch,
        prediction_type=prediction_type,
    )


def _pack_inference_block(
    clean_history_latents: torch.Tensor,
    block_latents: torch.Tensor,
    *,
    block_start: int,
    block_size: int,
) -> _PackedInferenceBlock:
    _validate_clean_history_and_block(clean_history_latents, block_latents)
    _validate_non_negative_int("block_start", block_start)
    _validate_positive_int("block_size", block_size)

    history_length = clean_history_latents.shape[1]
    if history_length != block_start:
        raise ValueError("clean_history_latents length must equal block_start")

    clean_history = clean_history_latents.detach()
    latents = torch.cat((clean_history, block_latents), dim=1)
    history_positions = torch.arange(
        block_start,
        device=block_latents.device,
        dtype=torch.long,
    )
    block_positions = torch.arange(
        block_start,
        block_start + block_latents.shape[1],
        device=block_latents.device,
        dtype=torch.long,
    )
    block_ids = torch.cat((history_positions, block_positions), dim=0) // block_size
    segment_ids = torch.cat(
        (
            torch.full(
                (block_start,),
                CLEAN_SEGMENT_ID,
                dtype=torch.long,
                device=block_latents.device,
            ),
            torch.full(
                (block_latents.shape[1],),
                NOISY_SEGMENT_ID,
                dtype=torch.long,
                device=block_latents.device,
            ),
        ),
        dim=0,
    )
    attention_mask = build_block_causal_attention_mask(block_ids, segment_ids)

    return _PackedInferenceBlock(
        latents=latents,
        attention_mask=attention_mask,
        block_ids=block_ids,
        segment_ids=segment_ids,
        current_slice=slice(block_start, block_start + block_latents.shape[1]),
    )


def _prediction_to_vector_field(
    *,
    prediction: torch.Tensor,
    current_latents: torch.Tensor,
    timestep: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    if prediction_type == "velocity":
        return prediction
    if prediction_type == "x0":
        timestep = timestep.reshape(timestep.shape[0], 1, 1).to(
            device=current_latents.device,
            dtype=current_latents.dtype,
        )
        eps = torch.finfo(current_latents.dtype).eps
        return (current_latents - prediction) / timestep.clamp_min(eps)
    raise ValueError("prediction_type must be 'velocity' or 'x0'")


def _denoising_time_grid(
    config: InferenceConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.linspace(
        config.diffusion.t_start,
        config.diffusion.t_end,
        steps=config.num_denoise_steps + 1,
        device=device,
        dtype=dtype,
    )


def _expand_timestep(
    timestep: torch.Tensor,
    *,
    batch_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return timestep.to(dtype=dtype).expand(batch_size)


def _cat_latents(
    prefix_latents: torch.Tensor,
    generated_blocks: list[torch.Tensor],
) -> torch.Tensor:
    if not generated_blocks:
        return prefix_latents
    return torch.cat((prefix_latents, *generated_blocks), dim=1)


def _initial_block_latents(
    prefix_latents: torch.Tensor,
    block: GenerationBlock,
    known_mask: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    block_latents = torch.zeros(
        prefix_latents.shape[0],
        block.end - block.start,
        prefix_latents.shape[-1],
        device=prefix_latents.device,
        dtype=prefix_latents.dtype,
    )
    unknown_mask = ~known_mask
    unknown_count = int(torch.count_nonzero(unknown_mask).item())
    if unknown_count:
        block_latents[:, unknown_mask, :] = torch.randn(
            (
                prefix_latents.shape[0],
                unknown_count,
                prefix_latents.shape[-1],
            ),
            generator=generator,
            device=prefix_latents.device,
            dtype=prefix_latents.dtype,
        )
    clean_block_latents = _known_prefix_block_latents(
        prefix_latents,
        block,
        block_latents,
    )
    return apply_clean_condition_repaint(
        block_latents,
        clean_block_latents,
        known_mask,
    )


def _known_prefix_block_latents(
    prefix_latents: torch.Tensor,
    block: GenerationBlock,
    reference_block_latents: torch.Tensor,
) -> torch.Tensor:
    clean_block_latents = torch.zeros_like(reference_block_latents)
    known_count = max(0, min(prefix_latents.shape[1], block.end) - block.start)
    if known_count:
        clean_block_latents[:, :known_count, :] = prefix_latents[
            :,
            block.start : block.start + known_count,
            :,
        ]
    return clean_block_latents


def _token_attention_to_transformer_mask(
    attention_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    return attention_mask[:, None, :].expand(
        attention_mask.shape[0],
        attention_mask.shape[1],
        attention_mask.shape[1],
    )


def _validate_prefix_token_ids(prefix_token_ids: torch.Tensor) -> None:
    if prefix_token_ids.dtype != torch.long:
        raise ValueError("prefix_token_ids must be a torch.long tensor")
    if prefix_token_ids.ndim != 2:
        raise ValueError("prefix_token_ids must be shaped [batch, prefix_len]")


def _validate_prefix_attention_mask(
    attention_mask: torch.Tensor,
    prefix_token_ids: torch.Tensor,
) -> None:
    if attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be a boolean tensor")
    if attention_mask.shape != prefix_token_ids.shape:
        raise ValueError("attention_mask must match prefix_token_ids shape")
    if attention_mask.device != prefix_token_ids.device:
        raise ValueError(
            "attention_mask and prefix_token_ids must be on the same device"
        )


def _validate_unconditional_prefix_token_ids(
    unconditional_prefix_token_ids: torch.Tensor,
    prefix_token_ids: torch.Tensor,
) -> None:
    _validate_prefix_token_ids(unconditional_prefix_token_ids)
    if unconditional_prefix_token_ids.shape != prefix_token_ids.shape:
        raise ValueError(
            "unconditional_prefix_token_ids must match prefix_token_ids shape"
        )
    if unconditional_prefix_token_ids.device != prefix_token_ids.device:
        raise ValueError(
            "unconditional_prefix_token_ids must be on the same device as "
            "prefix_token_ids"
        )


def _build_decoder_token_ids(
    prefix_token_ids: torch.Tensor,
    *,
    num_generated_tokens: int,
    fill_token_id: int,
    vocab_size: int,
) -> torch.Tensor:
    _validate_decoder_fill_token_id(fill_token_id, vocab_size)
    generated_token_ids = torch.full(
        (prefix_token_ids.shape[0], num_generated_tokens),
        fill_token_id,
        dtype=prefix_token_ids.dtype,
        device=prefix_token_ids.device,
    )
    return torch.cat((prefix_token_ids, generated_token_ids), dim=1)


def _build_decoder_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    num_generated_tokens: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None

    generated_mask = torch.ones(
        (attention_mask.shape[0], num_generated_tokens),
        dtype=torch.bool,
        device=attention_mask.device,
    )
    return _token_attention_to_transformer_mask(
        torch.cat((attention_mask, generated_mask), dim=1)
    )


def _validate_decoder_fill_token_id(fill_token_id: int, vocab_size: int) -> None:
    if isinstance(fill_token_id, bool) or not isinstance(fill_token_id, int):
        raise TypeError("decoder_fill_token_id must be an integer")
    if fill_token_id < 0 or fill_token_id >= vocab_size:
        raise ValueError("decoder_fill_token_id must be inside the decoder vocabulary")


def _validate_clean_condition_strategy(condition_strategy: str) -> None:
    if condition_strategy != "clean_condition_repaint":
        raise ValueError(
            "condition_strategy currently only supports "
            "'clean_condition_repaint'"
        )


def _validate_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_repaint_latents(
    block_latents: torch.Tensor,
    clean_block_latents: torch.Tensor,
) -> None:
    if block_latents.ndim != 3:
        raise ValueError(
            "block_latents must be shaped [batch, block_len, latent_dim]"
        )
    if clean_block_latents.shape != block_latents.shape:
        raise ValueError("clean_block_latents must match block_latents shape")
    if clean_block_latents.device != block_latents.device:
        raise ValueError(
            "clean_block_latents and block_latents must be on the same device"
        )
    if clean_block_latents.dtype != block_latents.dtype:
        raise ValueError("clean_block_latents and block_latents must share dtype")
    if not block_latents.is_floating_point():
        raise ValueError("block_latents must be a floating point tensor")


def _expand_known_mask(
    known_mask: torch.Tensor,
    block_latents: torch.Tensor,
) -> torch.Tensor:
    if known_mask.dtype != torch.bool:
        raise ValueError("known_mask must be a boolean tensor")
    if known_mask.device != block_latents.device:
        raise ValueError("known_mask and block_latents must be on the same device")

    batch_size, block_length, _ = block_latents.shape
    if known_mask.ndim == 1:
        if known_mask.shape != (block_length,):
            raise ValueError("known_mask must match block length")
        return known_mask.view(1, block_length, 1)
    if known_mask.ndim == 2:
        if known_mask.shape != (batch_size, block_length):
            raise ValueError("known_mask must match batch and block length")
        return known_mask.unsqueeze(-1)

    raise ValueError("known_mask must be shaped [block_len] or [batch, block_len]")


def _validate_sampler_inputs(
    *,
    dit: BlockCausalTextDiT,
    prefix_latents: torch.Tensor,
    unconditional_prefix_latents: torch.Tensor | None,
) -> None:
    if not hasattr(dit, "config") or not callable(dit):
        raise TypeError("dit must be a callable DiT-like object with a config")
    _validate_prefix_latents(prefix_latents, dit.config.latent_dim, "prefix_latents")
    if unconditional_prefix_latents is not None:
        _validate_prefix_latents(
            unconditional_prefix_latents,
            dit.config.latent_dim,
            "unconditional_prefix_latents",
        )
        if unconditional_prefix_latents.shape != prefix_latents.shape:
            raise ValueError(
                "unconditional_prefix_latents must match prefix_latents shape"
            )
        if unconditional_prefix_latents.device != prefix_latents.device:
            raise ValueError(
                "unconditional_prefix_latents must be on the same device as "
                "prefix_latents"
            )
        if unconditional_prefix_latents.dtype != prefix_latents.dtype:
            raise ValueError(
                "unconditional_prefix_latents must share prefix_latents dtype"
            )


def _validate_prefix_latents(
    prefix_latents: torch.Tensor,
    latent_dim: int,
    name: str,
) -> None:
    if not isinstance(prefix_latents, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if prefix_latents.ndim != 3:
        raise ValueError(f"{name} must be shaped [batch, prefix_len, latent_dim]")
    if not prefix_latents.is_floating_point():
        raise ValueError(f"{name} must be a floating point tensor")
    if prefix_latents.shape[-1] != latent_dim:
        raise ValueError(f"{name} latent dimension must match dit.config.latent_dim")


def _validate_generation_fits_dit(
    *,
    prefix_length: int,
    num_new_latents: int,
    sequence_length: int,
) -> None:
    if prefix_length + num_new_latents > sequence_length:
        raise ValueError(
            "prefix_length + num_new_latents must be no larger than "
            "dit.config.sequence_length"
        )


def _validate_clean_history_and_block(
    clean_history_latents: torch.Tensor,
    block_latents: torch.Tensor,
) -> None:
    _validate_repaint_latents(
        block_latents,
        torch.zeros_like(block_latents),
    )
    if clean_history_latents.ndim != 3:
        raise ValueError(
            "clean_history_latents must be shaped [batch, history_len, latent_dim]"
        )
    if clean_history_latents.shape[0] != block_latents.shape[0]:
        raise ValueError("clean_history_latents batch size must match block_latents")
    if clean_history_latents.shape[-1] != block_latents.shape[-1]:
        raise ValueError("clean_history_latents latent dim must match block_latents")
    if clean_history_latents.device != block_latents.device:
        raise ValueError("clean_history_latents must be on the block_latents device")
    if clean_history_latents.dtype != block_latents.dtype:
        raise ValueError("clean_history_latents must share block_latents dtype")
    if not clean_history_latents.is_floating_point():
        raise ValueError("clean_history_latents must be a floating point tensor")


def _validate_matching_vector_fields(
    unconditional_field: torch.Tensor,
    conditional_field: torch.Tensor,
) -> None:
    _validate_repaint_latents(conditional_field, unconditional_field)


def _resolve_positive_int(name: str, value: int | None, default: int) -> int:
    resolved = default if value is None else value
    _validate_positive_int(name, resolved)
    return resolved


def _resolve_non_negative_float(
    name: str,
    value: float | None,
    default: float,
) -> float:
    resolved = default if value is None else value
    return _validate_non_negative_real(name, resolved)


def _validate_non_negative_real(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


__all__ = (
    "GenerationBlock",
    "InferenceOutput",
    "apply_clean_condition_repaint",
    "combine_cfg_vector_fields",
    "encode_prefix_latents",
    "generate",
    "iter_generation_blocks",
    "sample_latent_blocks",
)
