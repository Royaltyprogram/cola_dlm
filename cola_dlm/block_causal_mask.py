"""Packed latent helpers for block-causal DiT inputs.

Inputs are latent tensors shaped ``[batch, L, latent_dim]``. Packed outputs
place clean history ``z0[:, : L - block_size]`` before noisy targets
``zt[:, :L]``; clean history is detached by default. Block ids are 0-indexed
throughout, and attention masks are boolean with ``True`` meaning the query is
allowed to attend to the key.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import torch


CLEAN_SEGMENT_ID = 0
NOISY_SEGMENT_ID = 1

__all__ = [
    "CLEAN_SEGMENT_ID",
    "NOISY_SEGMENT_ID",
    "PackedDiTInputs",
    "build_block_causal_attention_mask",
    "build_packed_dit_inputs",
]


@dataclass(frozen=True)
class PackedDiTInputs:
    """Packed tensors and metadata consumed by the future DiT backbone."""

    latents: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    block_ids: torch.Tensor
    segment_ids: torch.Tensor


def build_packed_dit_inputs(
    z0: torch.Tensor,
    zt: torch.Tensor,
    block_size: int,
    *,
    detach_clean_context: bool = True,
) -> PackedDiTInputs:
    """Pack clean history and noisy targets for block-causal DiT attention.

    ``z0`` and ``zt`` must be floating tensors shaped ``[batch, L, latent_dim]``
    with matching shape, dtype, and device. The packed latent order is clean
    history first, then noisy targets. ``block_ids`` are 0-indexed,
    ``attention_mask[query, key] == True`` means attention is allowed, and clean
    history is detached by default.
    """

    _validate_latent_pair(z0, zt)
    block_size = _normalize_block_size(block_size, z0.shape[1])

    batch_size, sequence_length, _ = z0.shape
    clean_length = sequence_length - block_size
    clean_context = z0[:, :clean_length]
    if detach_clean_context:
        clean_context = clean_context.detach()
    noisy_targets = zt[:, :sequence_length]
    latents = torch.cat((clean_context, noisy_targets), dim=1)

    clean_block_ids = torch.arange(clean_length, device=z0.device) // block_size
    noisy_block_ids = torch.arange(sequence_length, device=z0.device) // block_size
    block_ids = torch.cat((clean_block_ids, noisy_block_ids), dim=0)

    segment_ids = torch.cat(
        (
            torch.full(
                (clean_length,),
                CLEAN_SEGMENT_ID,
                dtype=torch.long,
                device=z0.device,
            ),
            torch.full(
                (sequence_length,),
                NOISY_SEGMENT_ID,
                dtype=torch.long,
                device=z0.device,
            ),
        ),
        dim=0,
    )
    loss_mask = torch.cat(
        (
            torch.zeros(
                batch_size,
                clean_length,
                dtype=torch.bool,
                device=z0.device,
            ),
            torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=z0.device,
            ),
        ),
        dim=1,
    )
    attention_mask = build_block_causal_attention_mask(block_ids, segment_ids)

    return PackedDiTInputs(
        latents=latents,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        block_ids=block_ids,
        segment_ids=segment_ids,
    )


def build_block_causal_attention_mask(
    block_ids: torch.Tensor,
    segment_ids: torch.Tensor,
) -> torch.Tensor:
    """Build a boolean block-causal mask from 0-indexed block and segment ids.

    Rows are query positions and columns are key positions. ``True`` means the
    query may attend to that key.
    """

    query_block_ids = block_ids[:, None]
    key_block_ids = block_ids[None, :]
    query_is_clean = segment_ids[:, None] == CLEAN_SEGMENT_ID
    query_is_noisy = segment_ids[:, None] == NOISY_SEGMENT_ID
    key_is_clean = segment_ids[None, :] == CLEAN_SEGMENT_ID
    key_is_noisy = segment_ids[None, :] == NOISY_SEGMENT_ID

    clean_query_visibility = (
        query_is_clean & key_is_clean & (key_block_ids <= query_block_ids)
    )
    noisy_query_clean_visibility = (
        query_is_noisy & key_is_clean & (key_block_ids < query_block_ids)
    )
    noisy_query_current_block_visibility = (
        query_is_noisy & key_is_noisy & (key_block_ids == query_block_ids)
    )

    return (
        clean_query_visibility
        | noisy_query_clean_visibility
        | noisy_query_current_block_visibility
    )


def _validate_latent_pair(z0: torch.Tensor, zt: torch.Tensor) -> None:
    if not isinstance(z0, torch.Tensor) or not isinstance(zt, torch.Tensor):
        raise TypeError("z0 and zt must be torch.Tensor instances")
    if z0.ndim != 3 or zt.ndim != 3:
        raise ValueError(
            "z0 and zt must be rank-3 tensors shaped [batch, L, latent_dim], "
            f"got z0.shape={tuple(z0.shape)} and zt.shape={tuple(zt.shape)}"
        )
    if z0.shape != zt.shape:
        raise ValueError(
            "z0 and zt must have matching shapes [batch, L, latent_dim], "
            f"got z0.shape={tuple(z0.shape)} and zt.shape={tuple(zt.shape)}"
        )
    if z0.device != zt.device:
        raise ValueError(
            "z0 and zt must be on the same device, "
            f"got {z0.device} and {zt.device}"
        )
    if not z0.dtype.is_floating_point or not zt.dtype.is_floating_point:
        raise TypeError(
            "z0 and zt must be floating point tensors, "
            f"got {z0.dtype!r} and {zt.dtype!r}"
        )
    if z0.dtype != zt.dtype:
        raise ValueError(
            "z0 and zt must have matching dtypes, "
            f"got {z0.dtype!r} and {zt.dtype!r}"
        )


def _normalize_block_size(block_size: int, sequence_length: int) -> int:
    if isinstance(block_size, bool):
        raise TypeError("block_size must be an integer, got bool")
    try:
        block_size = operator.index(block_size)
    except TypeError as exc:
        raise TypeError("block_size must be an integer") from exc

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size!r}")
    if block_size > sequence_length:
        raise ValueError(
            "block_size must be less than or equal to sequence length L, "
            f"got block_size={block_size} and L={sequence_length}"
        )
    if sequence_length % block_size != 0:
        raise ValueError(
            "sequence length must be divisible by block_size, "
            f"got L={sequence_length} and block_size={block_size}"
        )
    return block_size
