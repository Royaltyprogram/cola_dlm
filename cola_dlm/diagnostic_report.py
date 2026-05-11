"""Compact Markdown diagnostics reports for tiny debug training runs."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from cola_dlm.diagnostics import render_packed_block_causal_attention_mask


_METRIC_ORDER = (
    "loss",
    "reconstruction_nll",
    "reconstruction_accuracy",
    "kl",
    "vae_loss",
    "flow_matching_loss",
    "reference_kl",
    "posterior_regularizer",
    "mask_loss",
    "logsnr",
    "latent_norm_mean",
    "latent_norm_std",
    "posterior_variance_mean",
    "posterior_variance_std",
)
_FLOW_BLOCK_PREFIX = "flow_matching_loss_block_"


def build_diagnostics_report(
    *,
    stage_name: str,
    metrics_record: Mapping[str, Any],
    block_loss_metrics: Mapping[str | int, Any] | None = None,
    attention_mask_text: str | None = None,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> str:
    """Return a deterministic compact Markdown diagnostics report."""

    if not stage_name:
        raise ValueError("stage_name must be non-empty")
    if not isinstance(metrics_record, Mapping):
        raise TypeError("metrics_record must be a mapping")

    lines = [f"# {stage_name} Diagnostics Report", "", "## Run"]
    if "step" in metrics_record:
        lines.append(f"- Final step: {_format_value(metrics_record['step'])}")
    if metrics_path is not None:
        lines.append(f"- Metrics file: `{metrics_path}`")
    if checkpoint_path is not None:
        lines.append(f"- Checkpoint: `{checkpoint_path}`")

    lines.extend(["", "## Key Metrics", "", "| Metric | Value |", "| --- | ---: |"])
    for name in _METRIC_ORDER:
        if name in metrics_record:
            lines.append(f"| `{name}` | {_format_value(metrics_record[name])} |")

    block_losses = _collect_block_losses(metrics_record, block_loss_metrics)
    if block_losses:
        lines.extend([
            "",
            "## Flow Matching By Block",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ])
        for name, value in block_losses:
            lines.append(f"| `{name}` | {_format_value(value)} |")

    if attention_mask_text is not None:
        lines.extend(["", "## Attention Mask", "", "```text"])
        lines.extend(attention_mask_text.rstrip().splitlines())
        lines.append("```")

    return "\n".join(lines) + "\n"


def write_diagnostics_report(
    path: str | Path,
    *,
    stage_name: str,
    metrics_record: Mapping[str, Any],
    block_loss_metrics: Mapping[str | int, Any] | None = None,
    attention_mask_text: str | None = None,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> Path:
    """Write a compact Markdown diagnostics report and return its path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_diagnostics_report(
        stage_name=stage_name,
        metrics_record=metrics_record,
        block_loss_metrics=block_loss_metrics,
        attention_mask_text=attention_mask_text,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )
    output_path.write_text(report, encoding="utf-8")
    return output_path


def render_attention_mask_for_report(
    *,
    sequence_length: int,
    block_size: int,
    max_packed_length: int = 32,
) -> str:
    """Return a rendered mask for small configs or a compact size note."""

    sequence_length = _normalize_positive_int(sequence_length, "sequence_length")
    block_size = _normalize_positive_int(block_size, "block_size")
    max_packed_length = _normalize_positive_int(
        max_packed_length,
        "max_packed_length",
    )
    if block_size > sequence_length:
        raise ValueError("block_size must be less than or equal to sequence_length")
    if sequence_length % block_size != 0:
        raise ValueError("sequence_length must be divisible by block_size")

    packed_length = sequence_length + (sequence_length - block_size)
    if packed_length > max_packed_length:
        return (
            "Attention mask not rendered because the expected packed length "
            f"{packed_length} exceeds {max_packed_length} positions "
            f"(sequence_length={sequence_length}, block_size={block_size})."
        )

    return render_packed_block_causal_attention_mask(sequence_length, block_size)


def _collect_block_losses(
    metrics_record: Mapping[str, Any],
    block_loss_metrics: Mapping[str | int, Any] | None,
) -> list[tuple[str, Any]]:
    source = metrics_record if block_loss_metrics is None else block_loss_metrics
    block_losses: list[tuple[int, str, Any]] = []
    for raw_name, value in source.items():
        normalized = _normalize_block_loss_name(raw_name)
        if normalized is None:
            continue
        block_index, name = normalized
        block_losses.append((block_index, name, value))
    return [
        (name, value)
        for _, name, value in sorted(block_losses, key=lambda item: item[0])
    ]


def _normalize_block_loss_name(name: str | int) -> tuple[int, str] | None:
    if isinstance(name, int) and not isinstance(name, bool):
        return name, f"{_FLOW_BLOCK_PREFIX}{name}"
    if not isinstance(name, str) or not name.startswith(_FLOW_BLOCK_PREFIX):
        return None

    block_text = name.removeprefix(_FLOW_BLOCK_PREFIX)
    if not block_text.isdecimal():
        return None
    return int(block_text), name


def _format_value(value: Any) -> str:
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        if detached.numel() != 1:
            raise ValueError("report metrics must be scalar values")
        value = detached.item()
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _normalize_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


__all__ = (
    "build_diagnostics_report",
    "render_attention_mask_for_report",
    "write_diagnostics_report",
)
