import pytest
import torch

import cola_dlm.block_causal_mask as block_causal_mask
from cola_dlm.block_causal_mask import (
    CLEAN_SEGMENT_ID,
    NOISY_SEGMENT_ID,
    build_packed_dit_inputs,
)


def test_build_packed_dit_inputs_preserves_shape_dtype_and_device():
    z0, zt = _tiny_latents()

    packed = build_packed_dit_inputs(z0, zt, block_size=2)

    assert packed.latents.shape == (1, 14, 3)
    assert packed.latents.dtype is z0.dtype
    assert packed.latents.device == z0.device
    assert packed.attention_mask.shape == (14, 14)
    assert packed.attention_mask.dtype is torch.bool
    assert packed.attention_mask.device == z0.device


def test_build_packed_dit_inputs_concatenates_clean_context_then_noisy_targets():
    z0, zt = _tiny_latents()

    packed = build_packed_dit_inputs(z0, zt, block_size=2)

    torch.testing.assert_close(packed.latents[:, :6], z0[:, :6])
    torch.testing.assert_close(packed.latents[:, 6:], zt[:, :8])


def test_build_packed_dit_inputs_marks_blocks_segments_and_loss_positions():
    z0, zt = _tiny_latents()

    packed = build_packed_dit_inputs(z0, zt, block_size=2)

    expected_block_ids = torch.tensor(
        [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 3, 3],
        device=z0.device,
    )
    expected_segment_ids = torch.tensor(
        [CLEAN_SEGMENT_ID] * 6 + [NOISY_SEGMENT_ID] * 8,
        device=z0.device,
    )
    expected_loss_mask = torch.tensor(
        [[False] * 6 + [True] * 8],
        device=z0.device,
    )

    assert torch.equal(packed.block_ids, expected_block_ids)
    assert torch.equal(packed.segment_ids, expected_segment_ids)
    assert torch.equal(packed.loss_mask, expected_loss_mask)


def test_build_packed_dit_inputs_detaches_clean_context_by_default():
    z0 = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3).requires_grad_()
    zt = torch.arange(100, 124, dtype=torch.float32).reshape(1, 8, 3).requires_grad_()

    packed = build_packed_dit_inputs(z0, zt, block_size=2)
    packed.latents.sum().backward()

    assert z0.grad is None
    torch.testing.assert_close(zt.grad, torch.ones_like(zt))


def test_attention_mask_allows_and_denies_noisy_query_pairs():
    z0, zt = _tiny_latents()

    packed = build_packed_dit_inputs(z0, zt, block_size=2)
    mask = packed.attention_mask

    # Packed positions for L=8, block_size=2:
    # clean indices 0..5 are clean blocks 0, 1, 2
    # noisy indices 6..13 are noisy blocks 0, 1, 2, 3
    for query_index in (6, 7):
        assert mask[query_index, 6:8].all()
        assert not mask[query_index, :6].any()
        assert not mask[query_index, 8:14].any()

    for query_index in (10, 11):
        assert mask[query_index, 0:4].all()
        assert mask[query_index, 10:12].all()
        assert not mask[query_index, 4:6].any()
        assert not mask[query_index, 6:8].any()
        assert not mask[query_index, 8:10].any()
        assert not mask[query_index, 12:14].any()

    assert mask[10, 11]
    assert mask[11, 10]


def test_attention_mask_keeps_clean_context_rows_clean_only_and_block_causal():
    z0, zt = _tiny_latents()

    packed = build_packed_dit_inputs(z0, zt, block_size=2)
    mask = packed.attention_mask
    clean_indices = slice(0, 6)
    noisy_indices = slice(6, 14)
    clean_block_ids = packed.block_ids[clean_indices]

    assert not mask[clean_indices, noisy_indices].any()
    for query_index in range(6):
        past_or_current_clean_keys = clean_block_ids <= packed.block_ids[query_index]
        future_clean_keys = clean_block_ids > packed.block_ids[query_index]
        assert mask[query_index, clean_indices][past_or_current_clean_keys].all()
        assert not mask[query_index, clean_indices][future_clean_keys].any()


def test_public_surface_lists_supported_imports():
    assert block_causal_mask.__all__ == [
        "CLEAN_SEGMENT_ID",
        "NOISY_SEGMENT_ID",
        "PackedDiTInputs",
        "build_block_causal_attention_mask",
        "build_packed_dit_inputs",
    ]


def test_build_packed_dit_inputs_rejects_invalid_rank():
    z0, zt = _tiny_latents()

    with pytest.raises(ValueError, match="rank-3"):
        build_packed_dit_inputs(z0[0], zt, block_size=2)


def test_build_packed_dit_inputs_rejects_mismatched_shapes():
    z0, zt = _tiny_latents()

    with pytest.raises(ValueError, match="matching shapes"):
        build_packed_dit_inputs(z0, zt[:, :-1], block_size=2)


def test_build_packed_dit_inputs_rejects_integer_latents():
    z0, zt = _tiny_latents()

    with pytest.raises(TypeError, match="floating point"):
        build_packed_dit_inputs(z0.to(torch.long), zt.to(torch.long), block_size=2)


@pytest.mark.parametrize("block_size", [0, -2])
def test_build_packed_dit_inputs_rejects_non_positive_block_size(block_size):
    z0, zt = _tiny_latents()

    with pytest.raises(ValueError, match="positive"):
        build_packed_dit_inputs(z0, zt, block_size=block_size)


def test_build_packed_dit_inputs_rejects_oversized_block_size():
    z0, zt = _tiny_latents()

    with pytest.raises(ValueError, match="less than or equal"):
        build_packed_dit_inputs(z0, zt, block_size=16)


def test_build_packed_dit_inputs_rejects_non_divisible_sequence_length():
    z0, zt = _tiny_latents()

    with pytest.raises(ValueError, match="divisible"):
        build_packed_dit_inputs(z0, zt, block_size=3)


def _tiny_latents() -> tuple[torch.Tensor, torch.Tensor]:
    z0 = torch.arange(24, dtype=torch.float64).reshape(1, 8, 3)
    zt = torch.arange(100, 124, dtype=torch.float64).reshape(1, 8, 3)
    return z0, zt
