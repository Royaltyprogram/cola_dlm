"""Lightweight latent projection export utilities."""

from __future__ import annotations

import csv
import importlib
import json
import operator
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cola_dlm.vae import TextVAEOutput


LatentBatch = torch.Tensor | TextVAEOutput


@dataclass(frozen=True)
class _PointMetadata:
    batch_index: int
    token_position: int
    token_id: int | None


def export_latent_projection(
    latents_or_outputs: LatentBatch | Iterable[LatentBatch],
    path: str | Path,
    *,
    max_points: int,
    token_ids: torch.Tensor | Iterable[torch.Tensor] | None = None,
    method: str = "pca",
    output_format: str | None = None,
) -> list[dict[str, Any]]:
    """Project latent points to 2D and write them as CSV or JSONL records.

    ``max_points`` is required so debug exports stay explicit and bounded.
    Records are emitted in batch-major, token-major order and truncated before
    projection.
    """

    max_points = _normalize_max_points(max_points)
    method = _normalize_method(method)
    output_path = Path(path)
    output_format = _normalize_output_format(output_path, output_format)

    latent_points, metadata = _collect_latent_points(
        latents_or_outputs,
        token_ids=token_ids,
        max_points=max_points,
    )
    if method == "pca":
        coordinates, explained_variance_ratio = _project_pca(latent_points)
        extra_fields: dict[str, float] = {
            "explained_variance_ratio_x": float(explained_variance_ratio[0].item()),
            "explained_variance_ratio_y": float(explained_variance_ratio[1].item()),
        }
    else:
        coordinates = _project_umap(latent_points)
        extra_fields = {}

    records = _build_projection_records(metadata, coordinates, extra_fields)
    _write_projection_records(output_path, records, output_format)
    return records


def _collect_latent_points(
    latents_or_outputs: LatentBatch | Iterable[LatentBatch],
    *,
    token_ids: torch.Tensor | Iterable[torch.Tensor] | None,
    max_points: int,
) -> tuple[torch.Tensor, list[_PointMetadata]]:
    batches = _normalize_latent_batches(latents_or_outputs)
    token_batches = _normalize_token_batches(token_ids, len(batches))

    points: list[torch.Tensor] = []
    metadata: list[_PointMetadata] = []
    batch_offset = 0
    latent_dim: int | None = None

    for batch, raw_token_ids in zip(batches, token_batches, strict=True):
        latents = _normalize_latent_tensor(_extract_latents(batch))
        if latent_dim is None:
            latent_dim = latents.shape[-1]
        elif latents.shape[-1] != latent_dim:
            raise ValueError("all latent batches must share the same latent_dim")

        normalized_token_ids = _normalize_token_ids(raw_token_ids, latents)
        for local_batch_index in range(latents.shape[0]):
            for token_position in range(latents.shape[1]):
                if len(points) >= max_points:
                    return torch.stack(points), metadata
                points.append(
                    latents[local_batch_index, token_position]
                    .detach()
                    .cpu()
                    .to(dtype=torch.float64)
                )
                token_id = None
                if normalized_token_ids is not None:
                    token_id = int(
                        normalized_token_ids[local_batch_index, token_position].item()
                    )
                metadata.append(
                    _PointMetadata(
                        batch_index=batch_offset + local_batch_index,
                        token_position=token_position,
                        token_id=token_id,
                    )
                )
        batch_offset += latents.shape[0]

    if not points:
        raise ValueError("latents must contain at least one point")
    return torch.stack(points), metadata


def _project_pca(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = points - points.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    components = _canonicalize_component_signs(vh[:2])
    coordinates = centered @ components.T

    if coordinates.shape[1] < 2:
        padding = coordinates.new_zeros(coordinates.shape[0], 2 - coordinates.shape[1])
        coordinates = torch.cat([coordinates, padding], dim=1)

    explained_variance = singular_values.square()
    total_variance = explained_variance.sum()
    if total_variance > 0:
        explained_ratio = explained_variance / total_variance
    else:
        explained_ratio = explained_variance.new_zeros(explained_variance.shape)
    if explained_ratio.shape[0] < 2:
        padding = explained_ratio.new_zeros(2 - explained_ratio.shape[0])
        explained_ratio = torch.cat([explained_ratio, padding])
    return coordinates[:, :2], explained_ratio[:2]


def _project_umap(points: torch.Tensor) -> torch.Tensor:
    try:
        umap_module = importlib.import_module("umap")
    except ImportError as exc:
        raise ImportError(
            "method='umap' requires the optional 'umap-learn' dependency; "
            "use method='pca', which is available without optional packages."
        ) from exc

    reducer_class = getattr(umap_module, "UMAP", None)
    if reducer_class is None:
        raise ImportError(
            "method='umap' requires umap.UMAP from the optional 'umap-learn' "
            "dependency; use method='pca' instead."
        )
    if points.shape[0] < 3:
        raise ValueError(
            "method='umap' requires at least three points; use method='pca' "
            "for tiny exports."
        )

    kwargs: dict[str, Any] = {"n_components": 2, "random_state": 0}
    kwargs["n_neighbors"] = min(15, points.shape[0] - 1)
    reducer = reducer_class(**kwargs)
    embedding = reducer.fit_transform(points.detach().cpu().tolist())
    coordinates = torch.as_tensor(embedding, dtype=torch.float64)
    if coordinates.shape != (points.shape[0], 2):
        raise RuntimeError("UMAP returned coordinates with an unexpected shape")
    return coordinates


def _canonicalize_component_signs(components: torch.Tensor) -> torch.Tensor:
    components = components.clone()
    for index in range(components.shape[0]):
        component = components[index]
        pivot = int(torch.argmax(component.abs()).item())
        if component[pivot] < 0:
            components[index] = -component
    return components


def _build_projection_records(
    metadata: list[_PointMetadata],
    coordinates: torch.Tensor,
    extra_fields: dict[str, float],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for point_metadata, point_coordinates in zip(metadata, coordinates, strict=True):
        record: dict[str, Any] = {
            "batch_index": point_metadata.batch_index,
            "token_position": point_metadata.token_position,
        }
        if point_metadata.token_id is not None:
            record["token_id"] = point_metadata.token_id
        record["x"] = float(point_coordinates[0].item())
        record["y"] = float(point_coordinates[1].item())
        record.update(extra_fields)
        records.append(record)
    return records


def _write_projection_records(
    path: Path,
    records: list[dict[str, Any]],
    output_format: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        path.write_text(
            "".join(
                json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        return

    fieldnames = [
        "batch_index",
        "token_position",
        *(("token_id",) if any("token_id" in record for record in records) else ()),
        "x",
        "y",
        *(
            ("explained_variance_ratio_x", "explained_variance_ratio_y")
            if "explained_variance_ratio_x" in records[0]
            else ()
        ),
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _normalize_latent_batches(
    latents_or_outputs: LatentBatch | Iterable[LatentBatch],
) -> list[LatentBatch]:
    if isinstance(latents_or_outputs, (torch.Tensor, TextVAEOutput)):
        return [latents_or_outputs]

    if not isinstance(latents_or_outputs, Iterable):
        raise TypeError("latents_or_outputs must be a tensor, TextVAEOutput, or iterable")
    batches = list(latents_or_outputs)
    if not batches:
        raise ValueError("latents_or_outputs must contain at least one batch")
    for batch in batches:
        if not isinstance(batch, (torch.Tensor, TextVAEOutput)):
            raise TypeError("each latent batch must be a tensor or TextVAEOutput")
    return batches


def _extract_latents(batch: LatentBatch) -> torch.Tensor:
    if isinstance(batch, TextVAEOutput):
        return batch.latents
    return batch


def _normalize_latent_tensor(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim == 2:
        latents = latents.unsqueeze(0)
    if latents.ndim != 3:
        raise ValueError("latent tensors must be shaped [batch, seq, latent]")
    if not latents.is_floating_point():
        raise TypeError("latent tensors must be floating point")
    if latents.shape[0] <= 0 or latents.shape[1] <= 0 or latents.shape[2] <= 0:
        raise ValueError("latent tensors must have non-empty dimensions")
    if not torch.isfinite(latents.detach()).all().item():
        raise ValueError("latent tensors must contain only finite values")
    return latents


def _normalize_token_batches(
    token_ids: torch.Tensor | Iterable[torch.Tensor] | None,
    num_batches: int,
) -> list[torch.Tensor | None]:
    if token_ids is None:
        return [None] * num_batches
    if isinstance(token_ids, torch.Tensor):
        if num_batches != 1:
            raise ValueError(
                "token_ids must be an iterable matching latent batches when "
                "latents_or_outputs contains multiple batches"
            )
        return [token_ids]

    if not isinstance(token_ids, Iterable):
        raise TypeError("token_ids must be a tensor, iterable of tensors, or None")
    token_batches = list(token_ids)
    if len(token_batches) != num_batches:
        raise ValueError("token_ids iterable must match the number of latent batches")
    for token_batch in token_batches:
        if not isinstance(token_batch, torch.Tensor):
            raise TypeError("each token_ids batch must be a tensor")
    return token_batches


def _normalize_token_ids(
    token_ids: torch.Tensor | None,
    latents: torch.Tensor,
) -> torch.Tensor | None:
    if token_ids is None:
        return None
    if token_ids.ndim == 1 and latents.shape[0] == 1:
        token_ids = token_ids.unsqueeze(0)
    if token_ids.shape != latents.shape[:2]:
        raise ValueError("token_ids must match latent batch and sequence dimensions")
    if token_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("token_ids must be an integer tensor")
    return token_ids.detach().cpu()


def _normalize_max_points(max_points: int) -> int:
    if isinstance(max_points, bool):
        raise TypeError("max_points must be an integer, got bool")
    try:
        max_points = operator.index(max_points)
    except TypeError as exc:
        raise TypeError("max_points must be an integer") from exc
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    return max_points


def _normalize_method(method: str) -> str:
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    method = method.lower()
    if method not in {"pca", "umap"}:
        raise ValueError("method must be 'pca' or 'umap'")
    return method


def _normalize_output_format(path: Path, output_format: str | None) -> str:
    if output_format is None:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return "csv"
        if suffix == ".jsonl":
            return "jsonl"
        raise ValueError("output_format is required unless path ends in .csv or .jsonl")
    if not isinstance(output_format, str):
        raise TypeError("output_format must be a string")
    output_format = output_format.lower()
    if output_format not in {"csv", "jsonl"}:
        raise ValueError("output_format must be 'csv' or 'jsonl'")
    return output_format


__all__ = ("export_latent_projection",)
