import torch

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


def _tiny_latents() -> tuple[torch.Tensor, torch.Tensor]:
    z0 = torch.arange(24, dtype=torch.float64).reshape(1, 8, 3)
    zt = torch.arange(100, 124, dtype=torch.float64).reshape(1, 8, 3)
    return z0, zt
