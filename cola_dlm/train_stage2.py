"""Command line training entrypoint for Stage 2 joint VAE-DiT training."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from cola_dlm.checkpointing import CheckpointError, load_checkpoint, save_checkpoint
from cola_dlm.config import OptimizerConfig, Stage2Config
from cola_dlm.config_io import (
    config_from_dict,
    config_to_dict,
    load_config,
    save_config,
)
from cola_dlm.dataset import TokenizedTextDataset
from cola_dlm.diagnostic_report import (
    render_attention_mask_for_report,
    write_diagnostics_report,
)
from cola_dlm.dit import BlockCausalTextDiT
from cola_dlm.logging import JSONLMetricsLogger
from cola_dlm.precision import bf16_autocast
from cola_dlm.stage2 import (
    create_frozen_reference_encoder,
    stage2_joint_training_step,
)
from cola_dlm.training_utils import (
    build_scheduler,
    cycle_batches,
    resolve_data_files,
    resolve_non_negative_int,
    resolve_positive_int,
    resolve_required,
)
from cola_dlm.vae import TextVAE, TextVAEEncoder


@dataclass(frozen=True)
class Stage2RunOptions:
    """Resolved run options for a local Stage 2 training job."""

    data_files: tuple[Path, ...]
    output_dir: Path
    max_steps: int
    batch_size: int
    checkpoint_every: int
    log_every: int
    device: torch.device
    seed: int


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Stage 2 joint trainer from command line arguments."""

    args = _build_parser().parse_args(argv)
    loaded = load_config(args.config, Stage2Config)
    options = _resolve_options(args, loaded.metadata)
    train(options=options, config=loaded.config, resume=args.resume)
    return 0


def train(
    *,
    options: Stage2RunOptions,
    config: Stage2Config,
    resume: str | Path | None = None,
) -> int:
    """Train Stage 2 until ``options.max_steps`` and return the final step."""

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
    batches = cycle_batches(dataloader)

    vae = TextVAE(config.vae).to(options.device)
    dit = BlockCausalTextDiT(config.dit).to(options.device)
    reference_encoder = create_frozen_reference_encoder(vae).to(options.device)
    optimizer = _build_optimizer(vae, dit, config.optimizer, config.vae_dit_lr_ratio)
    scheduler = build_scheduler(
        optimizer,
        config.optimizer,
        max_steps=options.max_steps,
    )

    global_step = 0
    if resume is not None:
        global_step, reference_encoder = _load_resume_checkpoint(
            resume,
            vae=vae,
            dit=dit,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=options.device,
        )

    checkpoint_config = _checkpoint_config(config, run_metadata)
    checkpoint_dir = options.output_dir / "checkpoints"
    log_path = options.output_dir / "metrics.jsonl"
    final_metrics_record: dict[str, Any] | None = None

    with JSONLMetricsLogger(log_path) as logger:
        while global_step < options.max_steps:
            batch = next(batches)
            token_ids = batch["input_ids"].to(options.device)
            attention_mask = batch["attention_mask"].to(options.device)
            step = global_step + 1
            lr_metrics = _learning_rates(optimizer)

            with bf16_autocast(
                options.device,
                enabled=config.optimizer.precision.lower() == "bf16",
            ):
                loss = stage2_joint_training_step(
                    vae,
                    reference_encoder,
                    dit,
                    optimizer,
                    token_ids,
                    attention_mask=attention_mask,
                    stage2_config=config,
                    max_grad_norm=config.optimizer.grad_clip,
                )

            scheduler.step()
            global_step = step
            metrics = loss.as_dict()
            final_metrics_record = {"step": step, **metrics, **lr_metrics}

            if step % options.log_every == 0:
                logger.log(step, {**metrics, **lr_metrics})

            if step % options.checkpoint_every == 0:
                _save_stage2_checkpoint(
                    checkpoint_dir / f"step_{step:08d}.pt",
                    vae=vae,
                    dit=dit,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    config=checkpoint_config,
                )

    final_checkpoint = checkpoint_dir / "final.pt"
    _save_stage2_checkpoint(
        final_checkpoint,
        vae=vae,
        dit=dit,
        optimizer=optimizer,
        scheduler=scheduler,
        step=global_step,
        config=checkpoint_config,
    )
    attention_mask_text = render_attention_mask_for_report(
        sequence_length=config.dit.sequence_length,
        block_size=config.dit.block_size,
    )
    write_diagnostics_report(
        options.output_dir / "diagnostics_report.md",
        stage_name="Stage 2",
        metrics_record=final_metrics_record or {"step": global_step},
        attention_mask_text=attention_mask_text,
        checkpoint_path=final_checkpoint,
        metrics_path=log_path,
    )
    return global_step


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Stage 2 VAE-DiT jointly on local token id files.",
    )
    parser.add_argument("--config", required=True, help="Path to a Stage 2 JSON recipe.")
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
) -> Stage2RunOptions:
    data_files = resolve_data_files(args.data, metadata)
    output_dir = Path(resolve_required(args.output_dir, metadata, "output_dir"))
    max_steps = resolve_positive_int(args.max_steps, metadata, "max_steps")
    batch_size = resolve_positive_int(args.batch_size, metadata, "batch_size")
    checkpoint_every = resolve_positive_int(
        args.checkpoint_every,
        metadata,
        "checkpoint_every",
    )
    log_every = resolve_positive_int(args.log_every, metadata, "log_every")
    seed = resolve_non_negative_int(args.seed, metadata, "seed", default=0)
    device = torch.device(args.device or metadata.get("device", "cpu"))
    return Stage2RunOptions(
        data_files=data_files,
        output_dir=output_dir,
        max_steps=max_steps,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
        log_every=log_every,
        device=device,
        seed=seed,
    )


def _build_optimizer(
    vae: TextVAE,
    dit: BlockCausalTextDiT,
    optimizer_config: OptimizerConfig,
    vae_dit_lr_ratio: float,
) -> torch.optim.Optimizer:
    if optimizer_config.name.lower() != "adamw":
        raise ValueError(
            f"Stage 2 CLI only supports AdamW, got {optimizer_config.name!r}"
        )

    vae_parameters = [
        parameter for parameter in vae.parameters() if parameter.requires_grad
    ]
    dit_parameters = [
        parameter for parameter in dit.parameters() if parameter.requires_grad
    ]
    if not vae_parameters:
        raise ValueError("VAE must expose at least one trainable parameter")
    if not dit_parameters:
        raise ValueError("DiT must expose at least one trainable parameter")

    if vae_dit_lr_ratio == 1.0:
        parameter_groups: Any = vae_parameters + dit_parameters
    else:
        parameter_groups = [
            {
                "params": vae_parameters,
                "lr": optimizer_config.peak_lr * vae_dit_lr_ratio,
            },
            {"params": dit_parameters, "lr": optimizer_config.peak_lr},
        ]

    return torch.optim.AdamW(
        parameter_groups,
        lr=optimizer_config.peak_lr,
        betas=optimizer_config.betas,
        weight_decay=optimizer_config.weight_decay,
    )


def _learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    if len(optimizer.param_groups) == 1:
        return {"lr": float(optimizer.param_groups[0]["lr"])}
    return {
        "vae_lr": float(optimizer.param_groups[0]["lr"]),
        "dit_lr": float(optimizer.param_groups[1]["lr"]),
    }


def _load_resume_checkpoint(
    path: str | Path,
    *,
    vae: TextVAE,
    dit: BlockCausalTextDiT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: Stage2Config,
    device: torch.device,
) -> tuple[int, TextVAEEncoder]:
    try:
        loaded = load_checkpoint(
            path,
            extra_models={"vae": vae, "dit": dit},
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
    except FileNotFoundError:
        raise
    except (CheckpointError, RuntimeError, ValueError) as exc:
        raise CheckpointError(f"Resume checkpoint is incompatible: {exc}") from exc

    try:
        checkpoint_config = config_from_dict(Stage2Config, loaded.config["config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(
            "Resume checkpoint is incompatible: missing Stage 2 config"
        ) from exc
    if checkpoint_config != config:
        raise CheckpointError(
            "Resume checkpoint is incompatible with the requested Stage 2 config"
        )

    reference_encoder = create_frozen_reference_encoder(vae).to(device)
    return loaded.step, reference_encoder


def _save_stage2_checkpoint(
    path: Path,
    *,
    vae: TextVAE,
    dit: BlockCausalTextDiT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: Mapping[str, Any],
) -> None:
    save_checkpoint(
        path,
        extra_models={"vae": vae, "dit": dit},
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        config=config,
        metadata={"stage": "stage2"},
    )


def _checkpoint_config(
    config: Stage2Config,
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {"config": config_to_dict(config), **dict(run_metadata)}


def _run_metadata(options: Stage2RunOptions) -> dict[str, Any]:
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
