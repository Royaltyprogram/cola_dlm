import torch

from cola_dlm.precision import as_fp32_loss, bf16_autocast, supports_bf16


def test_bf16_autocast_is_safe_on_cpu_without_cuda():
    with bf16_autocast("cpu"):
        values = torch.ones(2, dtype=torch.float32)
        result = values + 1.0

    assert supports_bf16("cpu") is False
    assert result.dtype == torch.float32


def test_as_fp32_loss_reduces_unreduced_loss_values():
    values = torch.tensor([1.0, 3.0], dtype=torch.bfloat16)

    loss = as_fp32_loss(values)

    assert loss.shape == torch.Size([])
    assert loss.dtype == torch.float32
    assert loss.item() == 2.0


def test_as_fp32_loss_preserves_scalar_shape():
    scalar = torch.tensor(2.0, dtype=torch.float16)

    loss = as_fp32_loss(scalar)

    assert loss.shape == torch.Size([])
    assert loss.dtype == torch.float32
    assert loss.item() == 2.0
