"""Datasets for already-tokenized local text files."""

from __future__ import annotations

from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import torch
from torch.utils.data import Dataset


PathInput = str | PathLike[str]


class TokenizedTextDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed-length examples from whitespace-separated token id files.

    Each input file must contain integer token ids separated by whitespace.
    The dataset does not tokenize text; it only chunks token ids that have
    already been produced by a tokenizer.
    """

    def __init__(
        self,
        paths: PathInput | Iterable[PathInput],
        *,
        sequence_length: int,
        stride: int | None = None,
        pad_token_id: int = 0,
        drop_last: bool = False,
    ) -> None:
        _validate_positive_integer("sequence_length", sequence_length)
        if stride is None:
            stride = sequence_length
        _validate_positive_integer("stride", stride)
        if stride > sequence_length:
            raise ValueError("stride must be no larger than sequence_length")
        _validate_non_negative_integer("pad_token_id", pad_token_id)

        self.paths = _normalize_paths(paths)
        self.sequence_length = sequence_length
        self.stride = stride
        self.pad_token_id = pad_token_id
        self.drop_last = drop_last

        tokens: list[int] = []
        for path in self.paths:
            tokens.extend(_read_tokenized_file(path))
        self._tokens = tokens
        self._chunks = _build_chunks(
            total_tokens=len(tokens),
            sequence_length=sequence_length,
            stride=stride,
            drop_last=drop_last,
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start, token_count = self._chunks[index]
        token_ids = self._tokens[start : start + token_count]
        attention_mask = [True] * token_count

        pad_count = self.sequence_length - token_count
        if pad_count:
            token_ids = token_ids + [self.pad_token_id] * pad_count
            attention_mask = attention_mask + [False] * pad_count

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        }


def _normalize_paths(paths: PathInput | Iterable[PathInput]) -> tuple[Path, ...]:
    if isinstance(paths, (str, PathLike)):
        normalized = (Path(paths),)
    else:
        normalized = tuple(Path(path) for path in paths)

    if not normalized:
        raise ValueError("paths must contain at least one tokenized text file")

    for path in normalized:
        if not path.exists():
            raise FileNotFoundError(f"tokenized text file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"tokenized text path must be a file: {path}")
    return normalized


def _read_tokenized_file(path: Path) -> list[int]:
    token_ids: list[int] = []
    with path.open(encoding="utf-8") as token_file:
        for line_number, line in enumerate(token_file, start=1):
            for raw_token_id in line.split():
                try:
                    token_id = int(raw_token_id)
                except ValueError as exc:
                    raise ValueError(
                        f"malformed token id {raw_token_id!r} in {path} "
                        f"on line {line_number}"
                    ) from exc
                if token_id < 0:
                    raise ValueError(
                        f"token ids must be non-negative integers; got "
                        f"{token_id!r} in {path} on line {line_number}"
                    )
                token_ids.append(token_id)
    return token_ids


def _build_chunks(
    *,
    total_tokens: int,
    sequence_length: int,
    stride: int,
    drop_last: bool,
) -> tuple[tuple[int, int], ...]:
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < total_tokens:
        end = start + sequence_length
        if end <= total_tokens:
            chunks.append((start, sequence_length))
            if end == total_tokens:
                break
        else:
            if not drop_last:
                chunks.append((start, total_tokens - start))
            break
        start += stride
    return tuple(chunks)


def _validate_positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_non_negative_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


__all__ = ("TokenizedTextDataset",)
