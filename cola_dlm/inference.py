"""Public inference outputs for block-wise Cola DLM generation."""

from dataclasses import dataclass
from typing import Any


def _reject_kv_cache(kv_cache: object | None) -> None:
    if kv_cache is not None:
        raise ValueError("kv_cache is not supported yet; pass None")


@dataclass(frozen=True)
class InferenceOutput:
    """Container returned by the inference path."""

    prefix_latents: Any
    generated_latents: Any
    all_latents: Any
    response_logits: Any
    response_token_ids: Any
    kv_cache: None = None

    def __post_init__(self) -> None:
        _reject_kv_cache(self.kv_cache)


__all__ = ("InferenceOutput",)
