"""Typed configuration objects for the Cola DLM reproduction."""

from dataclasses import dataclass, field
from typing import Literal


PredictionType = Literal["velocity", "x0"]
TimestepSchedule = Literal["logit_normal", "uniform"]
ConditionStrategy = Literal[
    "clean_condition_repaint",
    "partial_repaint",
    "left_pad",
    "right_pad",
]


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _require_non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _require_optional_non_negative(name: str, value: int | float | None) -> None:
    if value is not None:
        _require_non_negative(name, value)


@dataclass
class VAEConfig:
    """Text VAE architecture and token-latent interface."""

    tokenizer_name: str = "OLMo 2"
    vocab_size: int = 100_278
    sequence_length: int = 512
    latent_dim: int = 16
    patch_size: int = 1
    encoder_layers: int = 4
    decoder_layers: int = 4
    hidden_size: int = 1_536
    ffn_size: int = 6_144
    num_attention_heads: int = 12
    attention_head_dim: int = 128
    dropout: float = 0.0
    activation: str = "gelu"
    use_rope: bool = True
    attention_pattern: str = "causal"

    def __post_init__(self) -> None:
        _require_positive("vocab_size", self.vocab_size)
        _require_positive("sequence_length", self.sequence_length)
        _require_positive("latent_dim", self.latent_dim)
        _require_positive("patch_size", self.patch_size)
        _require_positive("encoder_layers", self.encoder_layers)
        _require_positive("decoder_layers", self.decoder_layers)
        _require_positive("hidden_size", self.hidden_size)
        _require_positive("ffn_size", self.ffn_size)
        _require_positive("num_attention_heads", self.num_attention_heads)
        _require_positive("attention_head_dim", self.attention_head_dim)
        _require_non_negative("dropout", self.dropout)

        if self.sequence_length % self.patch_size != 0:
            raise ValueError("sequence_length must be divisible by patch_size")
        if self.attention_pattern != "causal":
            raise ValueError("attention_pattern must be 'causal'")
        if self.hidden_size != self.num_attention_heads * self.attention_head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * attention_head_dim"
            )
        if self.dropout >= 1:
            raise ValueError("dropout must be less than 1")
        if self.activation not in ("gelu", "silu"):
            raise ValueError("activation must be 'gelu' or 'silu'")
        if self.use_rope and self.attention_head_dim % 2 != 0:
            raise ValueError("attention_head_dim must be even when use_rope=True")


@dataclass
class DiTConfig:
    """Block-causal DiT prior configuration."""

    sequence_length: int = 512
    latent_dim: int = 16
    block_size: int = 16
    num_layers: int = 24
    hidden_size: int = 2_048
    ffn_size: int = 8_192
    num_attention_heads: int = 16
    attention_head_dim: int = 128
    positional_encoding: str = "rope"
    attention_pattern: str = "block_causal"

    def __post_init__(self) -> None:
        _require_positive("sequence_length", self.sequence_length)
        _require_positive("latent_dim", self.latent_dim)
        _require_positive("block_size", self.block_size)
        _require_positive("num_layers", self.num_layers)
        _require_positive("hidden_size", self.hidden_size)
        _require_positive("ffn_size", self.ffn_size)
        _require_positive("num_attention_heads", self.num_attention_heads)
        _require_positive("attention_head_dim", self.attention_head_dim)

        if self.block_size > self.sequence_length:
            raise ValueError("block_size must be no larger than sequence_length")
        if self.sequence_length % self.block_size != 0:
            raise ValueError("sequence_length must be divisible by block_size")
        if self.hidden_size != self.num_attention_heads * self.attention_head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * attention_head_dim")


@dataclass
class DiffusionConfig:
    """Flow Matching and timestep sampling configuration."""

    prediction_type: PredictionType = "velocity"
    timestep_schedule: TimestepSchedule = "logit_normal"
    logit_normal_loc: float = 1.0
    logit_normal_scale: float | None = None
    t_start: float = 1.0
    t_end: float = 0.0

    def __post_init__(self) -> None:
        if self.prediction_type not in ("velocity", "x0"):
            raise ValueError("prediction_type must be 'velocity' or 'x0'")
        if self.timestep_schedule not in ("logit_normal", "uniform"):
            raise ValueError("timestep_schedule must be 'logit_normal' or 'uniform'")
        _require_optional_non_negative("logit_normal_scale", self.logit_normal_scale)
        _require_non_negative("t_start", self.t_start)
        _require_non_negative("t_end", self.t_end)


@dataclass
class OptimizerConfig:
    """Paper-scale AdamW and learning-rate schedule defaults."""

    name: str = "adamw"
    peak_lr: float = 1.5e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 5_000
    warmup_start_lr: float = 1.0e-6
    min_lr: float = 1.0e-5
    lr_schedule: str = "cosine"
    precision: str = "bf16"

    def __post_init__(self) -> None:
        _require_positive("peak_lr", self.peak_lr)
        _require_non_negative("weight_decay", self.weight_decay)
        _require_positive("grad_clip", self.grad_clip)
        _require_non_negative("warmup_steps", self.warmup_steps)
        _require_non_negative("warmup_start_lr", self.warmup_start_lr)
        _require_non_negative("min_lr", self.min_lr)

        if len(self.betas) != 2:
            raise ValueError("betas must contain exactly two values")
        beta1, beta2 = self.betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must be in the range [0, 1)")


@dataclass
class Stage1Config:
    """Text VAE pretraining configuration."""

    vae: VAEConfig = field(default_factory=VAEConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    global_batch_size: int = 1_408
    tokens_per_step: int = 720_896
    kl_weight: float | None = None
    mask_loss_weight: float | None = None

    def __post_init__(self) -> None:
        _require_positive("global_batch_size", self.global_batch_size)
        _require_positive("tokens_per_step", self.tokens_per_step)
        _require_optional_non_negative("kl_weight", self.kl_weight)
        _require_optional_non_negative("mask_loss_weight", self.mask_loss_weight)


@dataclass
class Stage2Config:
    """Joint VAE and block-causal DiT training configuration."""

    vae: VAEConfig = field(default_factory=VAEConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    global_batch_size: int = 1_408
    tokens_per_step: int = 720_896
    vae_dit_lr_ratio: float = 1.0
    vae_loss_weight: float | None = None
    flow_matching_loss_weight: float | None = None
    reference_kl_weight: float | None = None

    def __post_init__(self) -> None:
        _require_positive("global_batch_size", self.global_batch_size)
        _require_positive("tokens_per_step", self.tokens_per_step)
        _require_positive("vae_dit_lr_ratio", self.vae_dit_lr_ratio)
        _require_optional_non_negative("vae_loss_weight", self.vae_loss_weight)
        _require_optional_non_negative(
            "flow_matching_loss_weight",
            self.flow_matching_loss_weight,
        )
        _require_optional_non_negative("reference_kl_weight", self.reference_kl_weight)
        _require_matching_latent_shape(self.vae, self.dit)


@dataclass
class InferenceConfig:
    """Prefix-conditioned block-wise generation defaults."""

    vae: VAEConfig = field(default_factory=VAEConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    denoising_steps: int = 16
    cfg_scale: float = 7.0
    max_new_tokens: int = 32
    condition_strategy: ConditionStrategy = "clean_condition_repaint"

    def __post_init__(self) -> None:
        _require_positive("denoising_steps", self.denoising_steps)
        _require_non_negative("cfg_scale", self.cfg_scale)
        _require_positive("max_new_tokens", self.max_new_tokens)
        if self.condition_strategy not in (
            "clean_condition_repaint",
            "partial_repaint",
            "left_pad",
            "right_pad",
        ):
            raise ValueError("condition_strategy must be a known first-block strategy")
        _require_matching_latent_shape(self.vae, self.dit)


def _require_matching_latent_shape(vae: VAEConfig, dit: DiTConfig) -> None:
    if vae.sequence_length != dit.sequence_length:
        raise ValueError("VAE and DiT sequence_length must match")
    if vae.latent_dim != dit.latent_dim:
        raise ValueError("VAE and DiT latent_dim must match")


__all__ = (
    "VAEConfig",
    "DiTConfig",
    "DiffusionConfig",
    "OptimizerConfig",
    "Stage1Config",
    "Stage2Config",
    "InferenceConfig",
)
