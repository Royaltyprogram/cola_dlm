"""Small precision helpers shared by training entrypoints."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch


def supports_bf16(device: str | torch.device) -> bool:
    """Return whether the selected device can use bf16 autocast safely."""

    selected_device = torch.device(device)
    if selected_device.type == "cuda":
        return bool(
            torch.cuda.is_available()
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
    if selected_device.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            return False
        is_supported = getattr(xpu, "is_bf16_supported", None)
        return bool(is_supported()) if callable(is_supported) else False
    return False


def bf16_autocast(
    device: str | torch.device,
    *,
    enabled: bool = True,
) -> ContextManager[None]:
    """Return a bf16 autocast context, or a no-op when bf16 is unavailable."""

    selected_device = torch.device(device)
    if enabled and supports_bf16(selected_device):
        return torch.autocast(
            device_type=selected_device.type,
            dtype=torch.bfloat16,
        )
    return nullcontext()


def as_fp32_loss(loss: torch.Tensor) -> torch.Tensor:
    """Return a scalar fp32 tensor suitable for backward calls.

    If a caller passes unreduced per-example or per-token loss values, this
    helper applies a mean reduction in fp32.
    """

    if not torch.is_tensor(loss):
        raise TypeError("loss must be a torch.Tensor")
    fp32_loss = loss.float()
    if fp32_loss.ndim == 0:
        return fp32_loss
    return fp32_loss.mean()


__all__ = (
    "as_fp32_loss",
    "bf16_autocast",
    "supports_bf16",
)
