"""Command line sampling entrypoint for local Cola DLM checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from cola_dlm.checkpointing import CheckpointError, load_checkpoint
from cola_dlm.config import InferenceConfig, Stage2Config
from cola_dlm.config_io import config_from_dict, load_config
from cola_dlm.dit import BlockCausalTextDiT
from cola_dlm.inference import generate
from cola_dlm.tokenizer import OfflineFallbackTokenizer, load_olmo2_tokenizer
from cola_dlm.vae import TextVAE


def main(argv: Sequence[str] | None = None) -> int:
    """Run local checkpoint sampling from command line arguments."""

    args = _build_parser().parse_args(argv)
    sample(args)
    return 0


def sample(args: argparse.Namespace) -> dict[str, Any]:
    """Load a checkpoint, generate response token ids, and write one record."""

    device = torch.device(args.device)
    if args.seed is not None:
        _seed_all(args.seed, device)

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config, config_source = _resolve_inference_config(
        checkpoint.config,
        config_path=args.config,
    )
    prompt_token_ids, prompt_metadata = _resolve_prompt_token_ids(args, config)

    vae = TextVAE(config.vae).to(device)
    dit = BlockCausalTextDiT(config.dit).to(device)
    try:
        load_checkpoint(
            args.checkpoint,
            extra_models={"vae": vae, "dit": dit},
            map_location=device,
        )
    except (CheckpointError, RuntimeError, ValueError) as exc:
        raise CheckpointError(f"Sample checkpoint is incompatible: {exc}") from exc

    vae.eval()
    dit.eval()
    prefix_token_ids = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(prefix_token_ids, dtype=torch.bool)
    generator = _build_generator(device, args.seed)

    with torch.no_grad():
        output = generate(
            vae,
            dit,
            prefix_token_ids,
            inference_config=config,
            max_new_tokens=args.max_new_tokens,
            attention_mask=attention_mask,
            generator=generator,
        )

    generated_token_ids = output.response_token_ids.detach().cpu().tolist()[0]
    record = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.step,
        "config_source": config_source,
        "config": None if args.config is None else str(args.config),
        "device": str(device),
        "seed": args.seed,
        "max_new_tokens": len(generated_token_ids),
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "all_token_ids": prompt_token_ids + generated_token_ids,
        **prompt_metadata,
    }
    _write_sample_output(args.output, record)
    return record


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample generated token ids from a local Stage 2 checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Stage 2 checkpoint path.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional InferenceConfig JSON recipe. Defaults to checkpoint config.",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        default=None,
        help=(
            "Text prompt. Uses the local tokenizer path with deterministic "
            "offline fallback, which is not OLMo-compatible."
        ),
    )
    prompt_group.add_argument(
        "--prompt-token-ids",
        default=None,
        help="Prompt token ids as JSON, comma-separated text, or whitespace text.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--output",
        required=True,
        help="Output .json or .jsonl path.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def _resolve_inference_config(
    checkpoint_config: Mapping[str, Any],
    *,
    config_path: str | Path | None,
) -> tuple[InferenceConfig, str]:
    if config_path is not None:
        return load_config(config_path, InferenceConfig).config, "override"

    config_values = checkpoint_config.get("config", checkpoint_config)
    if not isinstance(config_values, Mapping):
        raise CheckpointError("checkpoint config must contain a config object")

    try:
        return config_from_dict(InferenceConfig, config_values), "checkpoint"
    except (TypeError, ValueError):
        pass

    try:
        stage2_config = config_from_dict(Stage2Config, config_values)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(
            "checkpoint config is not compatible with Stage 2 or inference sampling"
        ) from exc

    return (
        InferenceConfig(
            vae=stage2_config.vae,
            dit=stage2_config.dit,
            diffusion=stage2_config.diffusion,
        ),
        "checkpoint",
    )


def _resolve_prompt_token_ids(
    args: argparse.Namespace,
    config: InferenceConfig,
) -> tuple[list[int], dict[str, Any]]:
    if args.prompt_token_ids is not None:
        token_ids = _parse_prompt_token_ids(args.prompt_token_ids)
        _validate_token_ids(token_ids, vocab_size=config.vae.vocab_size)
        return token_ids, {"prompt_source": "token_ids"}

    tokenizer = load_olmo2_tokenizer(
        config.vae.tokenizer_name,
        local_files_only=True,
        allow_fallback=True,
    )
    token_ids = tokenizer.encode(args.prompt)
    _validate_token_ids(token_ids, vocab_size=config.vae.vocab_size)
    tokenizer_kind = (
        "offline_fallback_not_olmo_compatible"
        if isinstance(tokenizer, OfflineFallbackTokenizer)
        else "huggingface"
    )
    return token_ids, {
        "prompt": args.prompt,
        "prompt_source": "text",
        "tokenizer": tokenizer_kind,
    }


def _parse_prompt_token_ids(value: str) -> list[int]:
    text = value.strip()
    if not text:
        raise ValueError("--prompt-token-ids must not be empty")

    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("--prompt-token-ids must be valid JSON") from exc
        if not isinstance(parsed, list):
            raise ValueError("--prompt-token-ids JSON must be a list")
        token_ids = parsed
    else:
        try:
            token_ids = [
                int(part)
                for part in re.split(r"[\s,]+", text)
                if part
            ]
        except ValueError as exc:
            raise ValueError(
                "--prompt-token-ids must contain only integers"
            ) from exc

    if not token_ids:
        raise ValueError("--prompt-token-ids must contain at least one token id")
    if any(
        not isinstance(token_id, int) or isinstance(token_id, bool)
        for token_id in token_ids
    ):
        raise ValueError("--prompt-token-ids must contain only integers")
    return list(token_ids)


def _validate_token_ids(token_ids: Sequence[int], *, vocab_size: int) -> None:
    if not token_ids:
        raise ValueError("prompt must contain at least one token id")
    for token_id in token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise ValueError("prompt token ids must be integers")
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(
                f"prompt token id {token_id} is outside vocab range [0, {vocab_size})"
            )


def _seed_all(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _build_generator(
    device: torch.device,
    seed: int | None,
) -> torch.Generator | None:
    if seed is None:
        return None
    if device.type not in {"cpu", "cuda"}:
        return None
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _write_sample_output(path: str | Path, record: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record), sort_keys=True)
    if output_path.suffix == ".jsonl":
        output_path.write_text(line + "\n", encoding="utf-8")
    else:
        output_path.write_text(
            json.dumps(dict(record), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    raise SystemExit(main())
