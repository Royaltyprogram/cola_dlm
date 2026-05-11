"""Lightweight metrics logging utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

import torch


class JSONLMetricsLogger:
    """Append one JSON object per training step, with optional TensorBoard."""

    def __init__(
        self,
        path: str | Path,
        *,
        tensorboard_dir: str | Path | None = None,
        flush: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self.path.open("a", encoding="utf-8")
        self._flush = flush
        self._writer = _create_tensorboard_writer(tensorboard_dir)

    def log(
        self,
        step: int,
        metrics: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Append metrics for one step."""

        _validate_step(step)
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        if "step" in metrics:
            raise ValueError("metrics must not contain the reserved key 'step'")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        json_metrics = {
            str(name): _to_jsonable(value)
            for name, value in metrics.items()
        }
        record = {"step": int(step), **json_metrics}
        if metadata is not None:
            record["metadata"] = _to_jsonable(dict(metadata))

        self._file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        if self._flush:
            self._file.flush()

        if self._writer is not None:
            for name, value in json_metrics.items():
                if _is_scalar_number(value):
                    self._writer.add_scalar(name, value, step)
            if self._flush:
                self._writer.flush()

    def close(self) -> None:
        """Flush and close any open logging resources."""

        if not self._file.closed:
            self._file.flush()
            self._file.close()
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> "JSONLMetricsLogger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _create_tensorboard_writer(tensorboard_dir: str | Path | None) -> Any | None:
    if tensorboard_dir is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    return SummaryWriter(log_dir=str(tensorboard_dir))


def _to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return _to_jsonable(detached.item())
        return detached.tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"Metric value of type {type(value).__name__} is not JSON serializable"
    )


def _is_scalar_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_step(step: int) -> None:
    if not isinstance(step, int) or isinstance(step, bool):
        raise TypeError("step must be an int")
    if step < 0:
        raise ValueError("step must be non-negative")


__all__ = ("JSONLMetricsLogger",)
