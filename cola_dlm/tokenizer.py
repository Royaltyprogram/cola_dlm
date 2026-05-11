"""Tokenizer boundary for OLMo 2 compatible text tokenizers.

The offline fallback in this module is deterministic and interface-compatible
for tests and smoke runs, but it does not reproduce OLMo 2 token ids.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


DEFAULT_OLMO2_TOKENIZER = "allenai/OLMo-2-1124-7B"


class TextTokenizer(Protocol):
    """Small tokenizer interface used by local evaluation helpers."""

    @property
    def pad_token_id(self) -> int:
        """Token id used for padding."""

    @property
    def eos_token_id(self) -> int:
        """Token id used to mark end of sequence."""

    @property
    def vocab_size(self) -> int:
        """Number of token ids exposed by the tokenizer."""

    def encode(self, text: str) -> list[int]:
        """Convert text to token ids."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Convert token ids back to text."""


@dataclass(frozen=True)
class HuggingFaceTokenizerAdapter:
    """Adapter that narrows a Hugging Face tokenizer to ``TextTokenizer``."""

    tokenizer: Any

    @property
    def pad_token_id(self) -> int:
        return _require_token_id(self.tokenizer, "pad_token_id")

    @property
    def eos_token_id(self) -> int:
        return _require_token_id(self.tokenizer, "eos_token_id")

    @property
    def vocab_size(self) -> int:
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        if vocab_size is None:
            try:
                vocab_size = len(self.tokenizer)
            except TypeError as exc:
                raise ValueError("tokenizer must expose vocab_size or __len__") from exc
        if (
            not isinstance(vocab_size, int)
            or isinstance(vocab_size, bool)
            or vocab_size <= 0
        ):
            raise ValueError("tokenizer vocab_size must be a positive integer")
        return vocab_size

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=False))


@dataclass(frozen=True)
class OfflineFallbackTokenizer:
    """Tiny UTF-8 byte tokenizer for deterministic offline tests.

    This tokenizer reserves ``0`` for padding and ``1`` for EOS, then maps each
    UTF-8 byte to ``byte + 2``. It is not suitable for reproducing OLMo 2 token
    ids or benchmark scores.
    """

    pad_token_id: int = 0
    eos_token_id: int = 1
    vocab_size: int = 258

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return [byte + 2 for byte in text.encode("utf-8")]

    def decode(self, token_ids: Sequence[int]) -> str:
        byte_values: list[int] = []
        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("token_ids must contain integers")
            if token_id in (self.pad_token_id, self.eos_token_id):
                continue
            if token_id < 2 or token_id >= self.vocab_size:
                raise ValueError(
                    "fallback token ids must be pad/eos or in the byte-token range"
                )
            byte_values.append(token_id - 2)

        try:
            return bytes(byte_values).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "fallback token ids do not form valid UTF-8 text"
            ) from exc


def load_olmo2_tokenizer(
    model_name_or_path: str = DEFAULT_OLMO2_TOKENIZER,
    *,
    local_files_only: bool = True,
    allow_fallback: bool = False,
) -> TextTokenizer:
    """Load an OLMo 2 compatible tokenizer without implicit network access.

    ``local_files_only`` defaults to ``True`` so tests and smoke runs do not
    trigger downloads. Set it to ``False`` only when network downloads are an
    intentional caller choice. When ``allow_fallback`` is true, a deterministic
    offline tokenizer is returned if the real tokenizer cannot be loaded.
    """

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        if allow_fallback:
            return OfflineFallbackTokenizer()
        raise RuntimeError(
            "transformers is not installed, so the OLMo 2 tokenizer cannot be "
            "loaded. Install cola-dlm[tokenizer] or transformers, or pass "
            "allow_fallback=True for the deterministic offline fallback."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        if allow_fallback:
            return OfflineFallbackTokenizer()
        raise RuntimeError(
            "could not load an OLMo 2 tokenizer from "
            f"{model_name_or_path!r} with local_files_only={local_files_only}. "
            "Provide a local tokenizer path, set local_files_only=False when "
            "downloads are acceptable, or pass allow_fallback=True for the "
            "deterministic offline fallback."
        ) from exc

    return HuggingFaceTokenizerAdapter(tokenizer)


def _require_token_id(tokenizer: Any, name: str) -> int:
    token_id = getattr(tokenizer, name, None)
    if not isinstance(token_id, int) or isinstance(token_id, bool):
        raise ValueError(f"tokenizer must expose integer {name}")
    return token_id


__all__ = (
    "DEFAULT_OLMO2_TOKENIZER",
    "HuggingFaceTokenizerAdapter",
    "OfflineFallbackTokenizer",
    "TextTokenizer",
    "load_olmo2_tokenizer",
)
