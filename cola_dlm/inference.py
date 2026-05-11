"""Public inference helpers for block-wise Cola DLM generation."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

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


__all__ = (
    "GenerationBlock",
    "InferenceOutput",
    "apply_clean_condition_repaint",
    "encode_prefix_latents",
    "iter_generation_blocks",
)
