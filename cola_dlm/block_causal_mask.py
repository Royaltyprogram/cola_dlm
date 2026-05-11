"""Packed latent helpers for block-causal DiT inputs."""

from __future__ import annotations

import operator
from dataclasses import dataclass

import torch


CLEAN_SEGMENT_ID = 0
NOISY_SEGMENT_ID = 1


@dataclass(frozen=True)
class PackedDiTInputs:
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
    _validate_latent_pair(z0, zt)
    block_size = _normalize_block_size(block_size, z0.shape[1])

    batch_size, sequence_length, _ = z0.shape
    clean_length = sequence_length - block_size
    packed_length = clean_length + sequence_length

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
    attention_mask = torch.ones(
        packed_length,
        packed_length,
        dtype=torch.bool,
        device=z0.device,
    )

    return PackedDiTInputs(
        latents=latents,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        block_ids=block_ids,
        segment_ids=segment_ids,
    )


def _validate_latent_pair(z0: torch.Tensor, zt: torch.Tensor) -> None:
    if not isinstance(z0, torch.Tensor) or not isinstance(zt, torch.Tensor):
        raise TypeError("z0 and zt must be torch.Tensor instances")
    if z0.ndim != 3 or zt.ndim != 3:
        raise ValueError("z0 and zt must be shaped [batch, L, latent_dim]")
    if z0.shape != zt.shape:
        raise ValueError(
            "z0 and zt must have matching shapes, "
            f"got {z0.shape} and {zt.shape}"
        )
    if z0.device != zt.device:
        raise ValueError(
            "z0 and zt must be on the same device, "
            f"got {z0.device} and {zt.device}"
        )
    if z0.dtype != zt.dtype:
        raise ValueError(
            "z0 and zt must have matching dtypes, "
            f"got {z0.dtype!r} and {zt.dtype!r}"
        )
    if not z0.dtype.is_floating_point or not zt.dtype.is_floating_point:
        raise TypeError(
            "z0 and zt must have floating point dtypes, "
            f"got {z0.dtype!r} and {zt.dtype!r}"
        )


def _normalize_block_size(block_size: int, sequence_length: int) -> int:
    if isinstance(block_size, bool):
        raise TypeError("block_size must be a positive integer")
    try:
        block_size = operator.index(block_size)
    except TypeError as exc:
        raise TypeError("block_size must be a positive integer") from exc

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size!r}")
    if block_size > sequence_length:
        raise ValueError(
            "block_size must be less than or equal to sequence length, "
            f"got block_size={block_size} and L={sequence_length}"
        )
    if sequence_length % block_size != 0:
        raise ValueError(
            "sequence length must be divisible by block_size, "
            f"got L={sequence_length} and block_size={block_size}"
        )
    return block_size
