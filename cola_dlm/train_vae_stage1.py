"""Command line training entrypoint for Stage 1 TextVAE pretraining."""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from cola_dlm.checkpointing import CheckpointError, load_checkpoint, save_checkpoint
from cola_dlm.config import OptimizerConfig, Stage1Config
from cola_dlm.config_io import (
    config_from_dict,
    config_to_dict,
    load_config,
    save_config,
)
from cola_dlm.dataset import TokenizedTextDataset
from cola_dlm.logging import JSONLMetricsLogger
from cola_dlm.precision import bf16_autocast
from cola_dlm.stage1 import stage1_pretraining_step
from cola_dlm.vae import TextVAE


@dataclass(frozen=True)
class Stage1RunOptions:
    """Resolved run options for a local Stage 1 training job."""

    data_files: tuple[Path, ...]
    output_dir: Path
    max_steps: int
    batch_size: int
    checkpoint_every: int
    log_every: int
    device: torch.device
    seed: int


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Stage 1 VAE trainer from command line arguments."""

    args = _build_parser().parse_args(argv)
    loaded = load_config(args.config, Stage1Config)
    options = _resolve_options(args, loaded.metadata)
    train(options=options, config=loaded.config, resume=args.resume)
    return 0


def train(
    *,
    options: Stage1RunOptions,
    config: Stage1Config,
    resume: str | Path | None = None,
) -> int:
    """Train Stage 1 until ``options.max_steps`` and return the final step."""

    torch.manual_seed(options.seed)
    if options.device.type == "cuda":
        torch.cuda.manual_seed_all(options.seed)

    dataset = TokenizedTextDataset(
        options.data_files,
        sequence_length=config.vae.sequence_length,
    )
    if len(dataset) == 0:
        raise ValueError("tokenized data did not produce any training examples")

    options.output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = _run_metadata(options)
    save_config(config, options.output_dir / "config.json", metadata=run_metadata)

    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False)
    batches = _cycle_batches(dataloader)

    model = TextVAE(config.vae).to(options.device)
    optimizer = _build_optimizer(model, config.optimizer)
    scheduler = _build_scheduler(
        optimizer,
        config.optimizer,
        max_steps=options.max_steps,
    )

    global_step = 0
    if resume is not None:
        global_step = _load_resume_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=options.device,
        )

    checkpoint_config = _checkpoint_config(config, run_metadata)
    checkpoint_dir = options.output_dir / "checkpoints"
    log_path = options.output_dir / "metrics.jsonl"

    with JSONLMetricsLogger(log_path) as logger:
        while global_step < options.max_steps:
            batch = next(batches)
            token_ids = batch["input_ids"].to(options.device)
            attention_mask = batch["attention_mask"].to(options.device)
            step = global_step + 1
            lr = optimizer.param_groups[0]["lr"]

            with bf16_autocast(
                options.device,
                enabled=config.optimizer.precision.lower() == "bf16",
            ):
                loss = stage1_pretraining_step(
                    model,
                    optimizer,
                    token_ids,
                    attention_mask=attention_mask,
                    stage1_config=config,
                    max_grad_norm=config.optimizer.grad_clip,
                )

            scheduler.step()
            global_step = step

            if step % options.log_every == 0:
                metrics = loss.as_dict()
                logger.log(step, {**metrics, "lr": lr})

            if step % options.checkpoint_every == 0:
                _save_stage1_checkpoint(
                    checkpoint_dir / f"step_{step:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    config=checkpoint_config,
                )

    _save_stage1_checkpoint(
        checkpoint_dir / "final.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=global_step,
        config=checkpoint_config,
    )
    return global_step


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Stage 1 TextVAE on local token id files.",
    )
    parser.add_argument("--config", required=True, help="Path to a Stage 1 JSON recipe.")
    parser.add_argument(
        "--data",
        action="extend",
        nargs="+",
        default=None,
        help="Whitespace-tokenized input file. Accepts one or more paths.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for run outputs.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def _resolve_options(
    args: argparse.Namespace,
    metadata: Mapping[str, Any],
) -> Stage1RunOptions:
    data_files = _resolve_data_files(args.data, metadata)
    output_dir = Path(_resolve_required(args.output_dir, metadata, "output_dir"))
    max_steps = _resolve_positive_int(args.max_steps, metadata, "max_steps")
    batch_size = _resolve_positive_int(args.batch_size, metadata, "batch_size")
    checkpoint_every = _resolve_positive_int(
        args.checkpoint_every,
        metadata,
        "checkpoint_every",
    )
    log_every = _resolve_positive_int(args.log_every, metadata, "log_every")
    seed = _resolve_non_negative_int(args.seed, metadata, "seed", default=0)
    device = torch.device(args.device or metadata.get("device", "cpu"))
    return Stage1RunOptions(
        data_files=data_files,
        output_dir=output_dir,
        max_steps=max_steps,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
        log_every=log_every,
        device=device,
        seed=seed,
    )


def _resolve_data_files(
    cli_values: list[str] | None,
    metadata: Mapping[str, Any],
) -> tuple[Path, ...]:
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


def _resolve_required(
    cli_value: Any,
    metadata: Mapping[str, Any],
    name: str,
) -> Any:
    value = cli_value if cli_value is not None else metadata.get(name)
    if value is None:
        raise ValueError(f"{name} must be provided by CLI or config metadata")
    return value


def _resolve_positive_int(
    cli_value: int | None,
    metadata: Mapping[str, Any],
    name: str,
) -> int:
    return _validate_int(
        name,
        _resolve_required(cli_value, metadata, name),
        minimum=1,
    )


def _resolve_non_negative_int(
    cli_value: int | None,
    metadata: Mapping[str, Any],
    name: str,
    *,
    default: int,
) -> int:
    value = cli_value if cli_value is not None else metadata.get(name, default)
    return _validate_int(name, value, minimum=0)


def _validate_int(name: str, value: Any, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _build_optimizer(
    model: torch.nn.Module,
    optimizer_config: OptimizerConfig,
) -> torch.optim.Optimizer:
    if optimizer_config.name.lower() != "adamw":
        raise ValueError(
            f"Stage 1 CLI only supports AdamW, got {optimizer_config.name!r}"
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.peak_lr,
        betas=optimizer_config.betas,
        weight_decay=optimizer_config.weight_decay,
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    optimizer_config: OptimizerConfig,
    *,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
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


def _cycle_batches(dataloader: DataLoader) -> Iterable[dict[str, torch.Tensor]]:
    while True:
        for batch in dataloader:
            yield batch


def _load_resume_checkpoint(
    path: str | Path,
    *,
    model: TextVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: Stage1Config,
    device: torch.device,
) -> int:
    try:
        loaded = load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
    except FileNotFoundError:
        raise
    except (CheckpointError, RuntimeError, ValueError) as exc:
        raise CheckpointError(f"Resume checkpoint is incompatible: {exc}") from exc

    try:
        checkpoint_config = config_from_dict(Stage1Config, loaded.config["config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(
            "Resume checkpoint is incompatible: missing Stage 1 config"
        ) from exc
    if checkpoint_config != config:
        raise CheckpointError(
            "Resume checkpoint is incompatible with the requested Stage 1 config"
        )
    return loaded.step


def _save_stage1_checkpoint(
    path: Path,
    *,
    model: TextVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: Mapping[str, Any],
) -> None:
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        config=config,
        metadata={"stage": "stage1"},
    )


def _checkpoint_config(
    config: Stage1Config,
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {"config": config_to_dict(config), **dict(run_metadata)}


def _run_metadata(options: Stage1RunOptions) -> dict[str, Any]:
    return {
        "data_files": [str(path) for path in options.data_files],
        "output_dir": str(options.output_dir),
        "max_steps": options.max_steps,
        "batch_size": options.batch_size,
        "checkpoint_every": options.checkpoint_every,
        "log_every": options.log_every,
        "device": str(options.device),
        "seed": options.seed,
    }


if __name__ == "__main__":
    raise SystemExit(main())
