"""Shared helpers for small local training entrypoints."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from cola_dlm.config import OptimizerConfig


def resolve_data_files(
    cli_values: list[str] | None,
    metadata: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Resolve tokenized data file paths from CLI arguments or recipe metadata."""

    values: Iterable[Any] | None = cli_values
    if values is None:
        metadata_values = metadata.get("data_files")
        if metadata_values is None:
            metadata_values = metadata.get("data")
        if isinstance(metadata_values, (str, Path)):
            values = [metadata_values]
        else:
            values = metadata_values
    if values is None:
        raise ValueError(
            "at least one tokenized data file is required via --data or data_files"
        )

    paths = tuple(Path(value) for value in values)
    if not paths:
        raise ValueError(
            "at least one tokenized data file is required via --data or data_files"
        )
    return paths


def resolve_required(
    cli_value: Any,
    metadata: Mapping[str, Any],
    name: str,
) -> Any:
    """Resolve a required run option from CLI arguments or recipe metadata."""

    value = cli_value if cli_value is not None else metadata.get(name)
    if value is None:
        raise ValueError(f"{name} must be provided by CLI or config metadata")
    return value


def resolve_positive_int(
    cli_value: int | None,
    metadata: Mapping[str, Any],
    name: str,
) -> int:
    """Resolve and validate a positive integer run option."""

    return validate_int(
        name,
        resolve_required(cli_value, metadata, name),
        minimum=1,
    )


def resolve_non_negative_int(
    cli_value: int | None,
    metadata: Mapping[str, Any],
    name: str,
    *,
    default: int,
) -> int:
    """Resolve and validate a non-negative integer run option."""

    value = cli_value if cli_value is not None else metadata.get(name, default)
    return validate_int(name, value, minimum=0)


def validate_int(name: str, value: Any, *, minimum: int) -> int:
    """Validate an integer lower bound and return the original integer."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    optimizer_config: OptimizerConfig,
    *,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build the warmup plus cosine scheduler shared by local trainers."""

    schedule_name = optimizer_config.lr_schedule.lower()
    peak_lr = optimizer_config.peak_lr
    warmup_steps = min(optimizer_config.warmup_steps, max_steps)
    start_factor = optimizer_config.warmup_start_lr / peak_lr
    min_factor = optimizer_config.min_lr / peak_lr

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            progress = step / warmup_steps
            return start_factor + (1.0 - start_factor) * progress
        if schedule_name != "cosine":
            return 1.0
        decay_steps = max(1, max_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def cycle_batches(dataloader: DataLoader) -> Iterable[dict[str, torch.Tensor]]:
    """Yield batches forever from a finite DataLoader."""

    while True:
        for batch in dataloader:
            yield batch


__all__ = (
    "build_scheduler",
    "cycle_batches",
    "resolve_data_files",
    "resolve_non_negative_int",
    "resolve_positive_int",
    "resolve_required",
    "validate_int",
)
