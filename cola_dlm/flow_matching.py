"""Standalone Flow Matching utilities."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence

import torch

from cola_dlm.config import DiffusionConfig


_DEFAULT_LOGIT_NORMAL_SCALE = 1.0


def sample_timestep(
    config: DiffusionConfig,
    shape: int | Sequence[int] | torch.Size | None = None,
    *,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
    num_discrete_timesteps: int | None = None,
) -> torch.Tensor:
    """Sample normalized timesteps from the configured schedule.

    LogitNormal sampling uses ``logit_normal_scale`` as sigma, defaulting to
    ``1.0`` when the config leaves the paper's unresolved scale as ``None``.
    When ``num_discrete_timesteps`` is provided, continuous samples are mapped
    to midpoint grid values ``(index + 0.5) / num_discrete_timesteps``.
    """

    sample_shape = _normalize_sample_shape(shape, batch_size)
    sample_dtype = _normalize_floating_dtype(dtype)
    schedule = config.timestep_schedule
    discrete_grid_size = _normalize_discrete_grid_size(num_discrete_timesteps)

    if schedule not in ("uniform", "logit_normal"):
        raise ValueError(
            "timestep_schedule must be 'uniform' or 'logit_normal', "
            f"got {schedule!r}"
        )

    if schedule == "uniform":
        timestep = _sample_uniform(sample_shape, device, sample_dtype, generator)
    else:
        timestep = _sample_logit_normal(
            sample_shape,
            config,
            device,
            sample_dtype,
            generator,
        )

    if discrete_grid_size is None:
        return timestep

    indices = torch.floor(timestep * discrete_grid_size).to(torch.long)
    indices = indices.clamp(max=discrete_grid_size - 1)
    return (indices.to(sample_dtype) + 0.5) / discrete_grid_size


def linear_bridge(
    z0: torch.Tensor,
    z1: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Interpolate clean latents ``z0`` toward base-noise latents ``z1``."""

    _validate_latent_pair(z0, z1)
    timestep = _broadcast_timestep(timestep, z0)
    return (1 - timestep) * z0 + timestep * z1


def velocity_target(z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    _validate_latent_pair(z0, z1)
    return z1 - z0


def x0_target(z0: torch.Tensor) -> torch.Tensor:
    _validate_floating_tensor(z0, "z0")
    return z0


def flow_matching_target(
    z0: torch.Tensor,
    z1: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    if prediction_type == "velocity":
        return velocity_target(z0, z1)
    if prediction_type == "x0":
        _validate_latent_pair(z0, z1)
        return x0_target(z0)
    raise ValueError(
        "prediction_type must be 'velocity' or 'x0', "
        f"got {prediction_type!r}"
    )


def _validate_latent_pair(z0: torch.Tensor, z1: torch.Tensor) -> None:
    if z0.shape != z1.shape:
        raise ValueError(
            "z0 and z1 must have matching shapes, "
            f"got {z0.shape} and {z1.shape}"
        )
    _validate_floating_tensor(z0, "z0")
    _validate_floating_tensor(z1, "z1")


def _validate_floating_tensor(tensor: torch.Tensor, name: str) -> None:
    if not tensor.dtype.is_floating_point:
        raise TypeError(
            f"{name} must have a floating point dtype, got {tensor.dtype!r}"
        )


def _broadcast_timestep(timestep: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    _validate_floating_tensor(timestep, "timestep")
    if timestep.ndim < latent.ndim and timestep.shape[:1] == latent.shape[:1]:
        timestep = timestep.reshape(
            *timestep.shape,
            *([1] * (latent.ndim - timestep.ndim)),
        )
    try:
        return torch.broadcast_to(timestep, latent.shape)
    except RuntimeError as exc:
        raise ValueError(
            "timestep must be broadcastable to the latent shape, "
            f"got {timestep.shape} for latent shape {latent.shape}"
        ) from exc


def _normalize_sample_shape(
    shape: int | Sequence[int] | torch.Size | None,
    batch_size: int | None,
) -> tuple[int, ...]:
    if shape is not None and batch_size is not None:
        raise ValueError("pass either shape or batch_size, not both")
    if batch_size is not None:
        return (_normalize_dim(batch_size, "batch_size"),)
    if shape is None:
        raise ValueError("shape or batch_size is required")
    if isinstance(shape, int):
        return (_normalize_dim(shape, "shape"),)
    if isinstance(shape, (str, bytes)):
        raise TypeError("shape must be an int or a sequence of ints")

    try:
        dims = tuple(_normalize_dim(dim, "shape dimension") for dim in shape)
    except TypeError as exc:
        raise TypeError("shape must be an int or a sequence of ints") from exc
    return dims


def _normalize_dim(value: object, name: str) -> int:
    try:
        dim = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc
    if dim < 0:
        raise ValueError(f"{name} must be non-negative, got {dim!r}")
    return dim


def _normalize_floating_dtype(dtype: torch.dtype | None) -> torch.dtype:
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be a floating point torch.dtype, got {dtype!r}")
    return dtype


def _normalize_discrete_grid_size(
    num_discrete_timesteps: int | None,
) -> int | None:
    if num_discrete_timesteps is None:
        return None
    grid_size = _normalize_dim(num_discrete_timesteps, "num_discrete_timesteps")
    if grid_size <= 0:
        raise ValueError(
            "num_discrete_timesteps must be positive, "
            f"got {num_discrete_timesteps!r}"
        )
    return grid_size


def _factory_kwargs(
    device: torch.device | str | None,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {"dtype": dtype}
    if device is not None:
        kwargs["device"] = device
    if generator is not None:
        kwargs["generator"] = generator
    return kwargs


def _sample_uniform(
    shape: tuple[int, ...],
    device: torch.device | str | None,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    timestep = torch.rand(shape, **_factory_kwargs(device, dtype, generator))
    return _clamp_open_unit(timestep, dtype)


def _sample_logit_normal(
    shape: tuple[int, ...],
    config: DiffusionConfig,
    device: torch.device | str | None,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    scale = (
        _DEFAULT_LOGIT_NORMAL_SCALE
        if config.logit_normal_scale is None
        else config.logit_normal_scale
    )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"logit_normal_scale must be positive, got {scale!r}")

    normal = torch.randn(shape, **_factory_kwargs(device, dtype, generator))
    return _clamp_open_unit(
        torch.sigmoid(normal * scale + config.logit_normal_loc),
        dtype,
    )


def _clamp_open_unit(timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    eps = torch.finfo(dtype).eps
    return timestep.clamp(min=eps, max=1.0 - eps)


__all__ = (
    "flow_matching_target",
    "linear_bridge",
    "sample_timestep",
    "velocity_target",
    "x0_target",
)
