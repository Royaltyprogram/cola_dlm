"""Stable checkpoint save/load helpers for local Cola DLM runs."""

from __future__ import annotations

import inspect
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cola_dlm import __version__


CHECKPOINT_FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be interpreted as a Cola DLM checkpoint."""


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Metadata returned after checkpoint state is loaded into caller objects."""

    step: int
    config: dict[str, Any]
    metadata: dict[str, Any]
    format_version: int
    package_version: str | None


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    extra_models: Mapping[str, torch.nn.Module] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    step: int,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save model, optimizer, scheduler, step, and config state.

    Object construction stays with the caller. This helper only serializes state
    dictionaries and later loads them into objects supplied by the caller.
    """

    _validate_step(step)
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    if model is not None and not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    named_models = _normalize_extra_models(extra_models)
    if model is None and not named_models:
        raise ValueError("checkpoint must include model or extra_models")

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "package_version": __version__,
        "model": None if model is None else model.state_dict(),
        "extra_models": {
            name: module.state_dict()
            for name, module in named_models.items()
        },
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "step": int(step),
        "config": dict(config),
        "metadata": dict(metadata or {}),
    }
    _atomic_torch_save(payload, Path(path))


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    extra_models: Mapping[str, torch.nn.Module] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> LoadedCheckpoint:
    """Load checkpoint state into caller-provided objects and return metadata."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    payload = _load_payload(checkpoint_path, map_location=map_location)
    _validate_payload(payload, checkpoint_path)

    if model is not None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        model_state = payload["model"]
        if model_state is None:
            raise CheckpointError("checkpoint does not contain model state")
        model.load_state_dict(model_state, strict=strict)

    named_models = _normalize_extra_models(extra_models)
    checkpoint_extra_models = payload["extra_models"]
    for name, module in named_models.items():
        if name not in checkpoint_extra_models:
            raise CheckpointError(
                f"checkpoint does not contain extra model state {name!r}"
            )
        module.load_state_dict(checkpoint_extra_models[name], strict=strict)

    if optimizer is not None:
        optimizer_state = payload["optimizer"]
        if optimizer_state is None:
            raise CheckpointError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(optimizer_state)

    if scheduler is not None:
        scheduler_state = payload["scheduler"]
        if scheduler_state is None:
            raise CheckpointError("checkpoint does not contain scheduler state")
        scheduler.load_state_dict(scheduler_state)

    return LoadedCheckpoint(
        step=payload["step"],
        config=dict(payload["config"]),
        metadata=dict(payload["metadata"]),
        format_version=payload["format_version"],
        package_version=payload["package_version"],
    )


def _normalize_extra_models(
    extra_models: Mapping[str, torch.nn.Module] | None,
) -> dict[str, torch.nn.Module]:
    if extra_models is None:
        return {}
    if not isinstance(extra_models, Mapping):
        raise TypeError("extra_models must be a mapping")

    normalized = {}
    for name, module in extra_models.items():
        if not isinstance(name, str) or not name:
            raise ValueError("extra model names must be non-empty strings")
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"extra model {name!r} must be a torch.nn.Module")
        normalized[name] = module
    return normalized


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_payload(
    path: Path,
    *,
    map_location: str | torch.device | None,
) -> Any:
    kwargs: dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    try:
        return torch.load(path, **kwargs)
    except Exception as exc:
        raise CheckpointError(f"Could not load checkpoint {path}: {exc}") from exc


def _validate_payload(payload: Any, path: Path) -> None:
    if not isinstance(payload, Mapping):
        raise CheckpointError(f"Checkpoint {path} must be a mapping")

    required_keys = {
        "format_version",
        "package_version",
        "model",
        "extra_models",
        "optimizer",
        "scheduler",
        "step",
        "config",
        "metadata",
    }
    missing_keys = sorted(required_keys - set(payload))
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise CheckpointError(
            f"Checkpoint {path} is missing required keys: {missing}"
        )

    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            "Unsupported checkpoint format version: "
            f"{payload['format_version']!r}"
        )
    if payload["package_version"] is not None and not isinstance(
        payload["package_version"],
        str,
    ):
        raise CheckpointError("checkpoint package_version must be a string or null")
    if payload["model"] is not None and not isinstance(payload["model"], Mapping):
        raise CheckpointError("checkpoint model state must be a mapping or null")
    if not isinstance(payload["extra_models"], Mapping):
        raise CheckpointError("checkpoint extra_models must be a mapping")
    for name, state in payload["extra_models"].items():
        if not isinstance(name, str) or not name:
            raise CheckpointError("checkpoint extra model names must be strings")
        if not isinstance(state, Mapping):
            raise CheckpointError(
                f"checkpoint extra model state {name!r} must be a mapping"
            )
    if payload["model"] is None and not payload["extra_models"]:
        raise CheckpointError("checkpoint must contain at least one model state")
    if payload["optimizer"] is not None and not isinstance(
        payload["optimizer"],
        Mapping,
    ):
        raise CheckpointError("checkpoint optimizer state must be a mapping or null")
    if payload["scheduler"] is not None and not isinstance(
        payload["scheduler"],
        Mapping,
    ):
        raise CheckpointError("checkpoint scheduler state must be a mapping or null")
    try:
        _validate_step(payload["step"])
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint step is invalid: {exc}") from exc
    if not isinstance(payload["config"], Mapping):
        raise CheckpointError("checkpoint config must be a mapping")
    if not isinstance(payload["metadata"], Mapping):
        raise CheckpointError("checkpoint metadata must be a mapping")


def _validate_step(step: int) -> None:
    if not isinstance(step, int) or isinstance(step, bool):
        raise TypeError("step must be an int")
    if step < 0:
        raise ValueError("step must be non-negative")


__all__ = (
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointError",
    "LoadedCheckpoint",
    "save_checkpoint",
    "load_checkpoint",
)
