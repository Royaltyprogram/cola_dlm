"""Parameter counting helpers and paper-scale report generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import nn

from cola_dlm.config import Stage1Config, Stage2Config
from cola_dlm.config_io import load_config
from cola_dlm.dit import BlockCausalTextDiT
from cola_dlm.transformer import OutputProjection
from cola_dlm.vae import TextVAE


PAPER_VAE_PARAMETERS = "about 500M"
PAPER_DIT_NON_EMBEDDING_PARAMETERS = "about 1.8B"
PAPER_COLA_DLM_EMBEDDING_PARAMETERS = 308_054_016
PAPER_BASELINE_EMBEDDING_PARAMETERS = 410_738_688


@dataclass(frozen=True)
class ParameterCounts:
    """Trainable and non-trainable parameter totals."""

    trainable: int
    non_trainable: int

    @property
    def total(self) -> int:
        return self.trainable + self.non_trainable


@dataclass(frozen=True)
class VAEComponentCounts:
    """Parameter totals for the local Text VAE implementation."""

    total: ParameterCounts
    encoder: ParameterCounts
    decoder: ParameterCounts
    embedding_parameters: int
    non_embedding_backbone_parameters: int
    encoder_embedding_parameters: int
    decoder_embedding_parameters: int
    decoder_output_projection_parameters: int


@dataclass(frozen=True)
class DiTComponentCounts:
    """Parameter totals for the local DiT implementation."""

    total: ParameterCounts
    embedding_parameters: int
    non_embedding_backbone_parameters: int
    input_projection_parameters: int
    timestep_embedding_parameters: int
    transformer_layer_parameters: int
    segment_embedding_parameters: int
    output_head_parameters: int


@dataclass(frozen=True)
class PaperParameterReport:
    """Resolved config paths, meta-device status, and component counts."""

    stage1_config_path: Path
    stage2_config_path: Path
    vae_device: str
    dit_device: str
    vae: VAEComponentCounts
    dit: DiTComponentCounts


def count_parameters(module: nn.Module) -> ParameterCounts:
    """Count trainable and non-trainable parameters in a module."""

    trainable = 0
    non_trainable = 0
    for parameter in module.parameters():
        count = parameter.numel()
        if parameter.requires_grad:
            trainable += count
        else:
            non_trainable += count
    return ParameterCounts(trainable=trainable, non_trainable=non_trainable)


def count_embedding_parameters(module: nn.Module) -> int:
    """Count parameters owned by ``nn.Embedding`` modules."""

    return _count_parameters_in_matching_modules(module, nn.Embedding)


def count_non_embedding_backbone_parameters(module: nn.Module) -> int:
    """Count all parameters except those owned by ``nn.Embedding`` modules."""

    return count_parameters(module).total - count_embedding_parameters(module)


def count_vae_components(model: TextVAE) -> VAEComponentCounts:
    """Count local Text VAE totals by encoder, decoder, and embedding bucket."""

    encoder = count_parameters(model.encoder)
    decoder = count_parameters(model.decoder)
    encoder_embedding_parameters = count_embedding_parameters(model.encoder)
    decoder_embedding_parameters = count_embedding_parameters(model.decoder)
    decoder_output_projection_parameters = _count_parameters_in_matching_modules(
        model.decoder,
        OutputProjection,
    )
    total = count_parameters(model)
    embedding_parameters = count_embedding_parameters(model)

    return VAEComponentCounts(
        total=total,
        encoder=encoder,
        decoder=decoder,
        embedding_parameters=embedding_parameters,
        non_embedding_backbone_parameters=total.total - embedding_parameters,
        encoder_embedding_parameters=encoder_embedding_parameters,
        decoder_embedding_parameters=decoder_embedding_parameters,
        decoder_output_projection_parameters=decoder_output_projection_parameters,
    )


def count_dit_components(model: BlockCausalTextDiT) -> DiTComponentCounts:
    """Count local DiT totals by major implementation component."""

    total = count_parameters(model)
    embedding_parameters = count_embedding_parameters(model)
    segment_embedding_parameters = (
        count_parameters(model.segment_embedding).total
        if model.segment_embedding is not None
        else 0
    )
    output_head_parameters = (
        count_parameters(model.output_norm).total
        + count_parameters(model.output_projection).total
    )

    return DiTComponentCounts(
        total=total,
        embedding_parameters=embedding_parameters,
        non_embedding_backbone_parameters=total.total - embedding_parameters,
        input_projection_parameters=count_parameters(model.input_projection).total,
        timestep_embedding_parameters=count_parameters(
            model.timestep_embedding,
        ).total,
        transformer_layer_parameters=count_parameters(model.layers).total,
        segment_embedding_parameters=segment_embedding_parameters,
        output_head_parameters=output_head_parameters,
    )


def meta_device_is_available() -> bool:
    """Return whether PyTorch can initialize modules directly on ``meta``."""

    try:
        with torch.device("meta"):
            probe = nn.Linear(1, 1)
    except Exception:
        return False
    return probe.weight.device.type == "meta"


def build_paper_parameter_report(
    stage2_config_path: str | Path,
    *,
    stage1_config_path: str | Path | None = None,
    prefer_meta: bool = True,
) -> PaperParameterReport:
    """Load paper recipes, instantiate models, and count their parameters."""

    stage2_path = Path(stage2_config_path)
    stage1_path = (
        Path(stage1_config_path)
        if stage1_config_path is not None
        else _default_stage1_config_path(stage2_path)
    )
    stage1 = load_config(stage1_path, Stage1Config)
    stage2 = load_config(stage2_path, Stage2Config)

    vae, vae_device = _instantiate_model(
        lambda: TextVAE(stage1.config.vae),
        prefer_meta=prefer_meta,
    )
    dit, dit_device = _instantiate_model(
        lambda: BlockCausalTextDiT(stage2.config.dit),
        prefer_meta=prefer_meta,
    )

    return PaperParameterReport(
        stage1_config_path=stage1_path,
        stage2_config_path=stage2_path,
        vae_device=vae_device,
        dit_device=dit_device,
        vae=count_vae_components(vae),
        dit=count_dit_components(dit),
    )


def render_markdown_report(report: PaperParameterReport) -> str:
    """Render a deterministic Markdown parameter-count report."""

    stage1_path = _display_path(report.stage1_config_path)
    stage2_path = _display_path(report.stage2_config_path)
    vae = report.vae
    dit = report.dit

    lines = [
        "# Cola DLM Parameter Counts",
        "",
        "Generated from checked-in paper-scale recipes. The CLI initializes "
        "models on the PyTorch `meta` device by default so this report can be "
        "produced without allocating paper-scale tensors.",
        "",
        "## Resolved Configs",
        "",
        f"- Stage 1 VAE config: `{stage1_path}`",
        f"- Stage 2 DiT config: `{stage2_path}`",
        f"- VAE initialization device: `{report.vae_device}`",
        f"- DiT initialization device: `{report.dit_device}`",
        "",
        "## Summary",
        "",
        "| Component | Local count | Paper reported value | Accounting note |",
        "| --- | ---: | --- | --- |",
        _summary_row(
            "Text VAE total",
            vae.total.total,
            PAPER_VAE_PARAMETERS,
            "Includes local decoder output projection.",
        ),
        _summary_row(
            "Text VAE token embeddings",
            vae.embedding_parameters,
            f"{PAPER_COLA_DLM_EMBEDDING_PARAMETERS:,} in the appendix",
            "Counts encoder and decoder token embeddings only.",
        ),
        _summary_row(
            "Text VAE decoder output projection",
            vae.decoder_output_projection_parameters,
            "not listed in appendix embedding row",
            "Counted with the local decoder and VAE total.",
        ),
        _summary_row(
            "DiT total",
            dit.total.total,
            PAPER_DIT_NON_EMBEDDING_PARAMETERS,
            "No `nn.Embedding` parameters are enabled in the paper config.",
        ),
        _summary_row(
            "DiT non-embedding backbone",
            dit.non_embedding_backbone_parameters,
            PAPER_DIT_NON_EMBEDDING_PARAMETERS,
            "Local transformer block is smaller than the reported paper scale.",
        ),
        "",
        "## VAE Components",
        "",
        "| Component | Trainable | Non-trainable | Total |",
        "| --- | ---: | ---: | ---: |",
        _parameter_row("Encoder", vae.encoder),
        _parameter_row("Decoder", vae.decoder),
        _parameter_row("Total", vae.total),
        "",
        "| VAE accounting bucket | Parameters |",
        "| --- | ---: |",
        _count_row("Encoder token embedding", vae.encoder_embedding_parameters),
        _count_row("Decoder token embedding", vae.decoder_embedding_parameters),
        _count_row(
            "Decoder output projection",
            vae.decoder_output_projection_parameters,
        ),
        _count_row("All token embeddings", vae.embedding_parameters),
        _count_row(
            "Non-embedding VAE parameters",
            vae.non_embedding_backbone_parameters,
        ),
        "",
        "## DiT Components",
        "",
        "| Component | Parameters |",
        "| --- | ---: |",
        _count_row("Input projection", dit.input_projection_parameters),
        _count_row("Timestep embedding MLP", dit.timestep_embedding_parameters),
        _count_row("Transformer layers", dit.transformer_layer_parameters),
        _count_row("Segment embedding", dit.segment_embedding_parameters),
        _count_row("Output norm and projection", dit.output_head_parameters),
        _count_row("All embeddings", dit.embedding_parameters),
        _count_row(
            "Non-embedding DiT parameters",
            dit.non_embedding_backbone_parameters,
        ),
        _count_row("Total DiT parameters", dit.total.total),
        "",
        "## Implementation Notes",
        "",
        "- The paper appendix reports Cola DLM embedding parameters as "
        f"`{PAPER_COLA_DLM_EMBEDDING_PARAMETERS:,}` and AR/LLaDA embedding "
        f"parameters as `{PAPER_BASELINE_EMBEDDING_PARAMETERS:,}`.",
        "- The local VAE has separate encoder and decoder token embeddings. Their "
        "sum matches the appendix Cola DLM embedding row.",
        "- The local decoder output projection is an untied vocabulary projection. "
        "It is not counted with embeddings here; it is counted with the decoder "
        "and with total VAE parameters.",
        "- The local DiT count is lower than the paper's `about 1.8B` "
        "non-embedding backbone report. This report records the current checked-in "
        "implementation instead of adjusting architecture constants to force a "
        "match.",
        "",
    ]
    return "\n".join(lines)


def write_markdown_report(
    output_path: str | Path,
    report: PaperParameterReport,
) -> None:
    """Write a rendered report to disk."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown_report(report), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Count Cola DLM parameters from checked-in configs.",
    )
    parser.add_argument(
        "--config",
        default="configs/stage2_paper.json",
        help="Stage 2 config recipe path.",
    )
    parser.add_argument(
        "--stage1-config",
        default=None,
        help="Stage 1 VAE config recipe path. Defaults to configs/stage1_paper.json.",
    )
    parser.add_argument(
        "--output",
        default="docs/reproduction/cola_dlm/parameter_counts.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="Instantiate on the default device. Intended only for tiny configs.",
    )
    args = parser.parse_args(argv)

    report = build_paper_parameter_report(
        args.config,
        stage1_config_path=args.stage1_config,
        prefer_meta=not args.no_meta,
    )
    write_markdown_report(args.output, report)
    return 0


def _instantiate_model(
    factory: Callable[[], nn.Module],
    *,
    prefer_meta: bool,
) -> tuple[nn.Module, str]:
    if not prefer_meta:
        return factory(), "cpu"
    if not meta_device_is_available():
        raise RuntimeError(
            "PyTorch meta-device initialization is not available. "
            "Use --no-meta only with small configs."
        )
    with torch.device("meta"):
        return factory(), "meta"


def _count_parameters_in_matching_modules(
    module: nn.Module,
    module_type: type[nn.Module],
) -> int:
    seen: set[int] = set()
    total = 0
    for child in module.modules():
        if isinstance(child, module_type):
            for parameter in child.parameters():
                parameter_id = id(parameter)
                if parameter_id not in seen:
                    seen.add(parameter_id)
                    total += parameter.numel()
    return total


def _default_stage1_config_path(stage2_config_path: Path) -> Path:
    sibling = stage2_config_path.parent / "stage1_paper.json"
    if sibling.exists():
        return sibling
    return Path("configs/stage1_paper.json")


def _display_path(path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return str(resolved.relative_to(Path.cwd().resolve(strict=False)))
    except ValueError:
        return str(resolved)


def _format_count(count: int) -> str:
    return f"{count:,} ({count / 1_000_000:.1f}M)"


def _parameter_row(name: str, counts: ParameterCounts) -> str:
    return (
        f"| {name} | {counts.trainable:,} | "
        f"{counts.non_trainable:,} | {_format_count(counts.total)} |"
    )


def _count_row(name: str, count: int) -> str:
    return f"| {name} | {_format_count(count)} |"


def _summary_row(
    name: str,
    local_count: int,
    paper_value: str,
    note: str,
) -> str:
    return f"| {name} | {_format_count(local_count)} | {paper_value} | {note} |"


if __name__ == "__main__":
    raise SystemExit(main())
