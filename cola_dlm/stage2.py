"""Stage 2 joint VAE-DiT training helpers."""

from __future__ import annotations

import copy

from cola_dlm.config import Stage2Config
from cola_dlm.dit import BlockCausalTextDiT
from cola_dlm.vae import TextVAE, TextVAEEncoder


def create_frozen_reference_encoder(vae: TextVAE) -> TextVAEEncoder:
    """Return an eval-mode, gradient-frozen copy of ``vae.encoder``."""

    if not isinstance(vae, TextVAE):
        raise TypeError("vae must be a TextVAE")

    reference_encoder = copy.deepcopy(vae.encoder)
    reference_encoder.eval()
    for parameter in reference_encoder.parameters():
        parameter.requires_grad_(False)
    return reference_encoder


def _validate_stage2_component_shapes(
    *,
    vae: TextVAE,
    reference_encoder: TextVAEEncoder,
    dit: BlockCausalTextDiT,
    stage2_config: Stage2Config,
) -> None:
    """Check module token-latent boundaries against the Stage 2 config."""

    _require_instance("vae", vae, TextVAE)
    _require_instance("reference_encoder", reference_encoder, TextVAEEncoder)
    _require_instance("dit", dit, BlockCausalTextDiT)
    _require_instance("stage2_config", stage2_config, Stage2Config)

    _require_equal(
        "vae.encoder.latent_dim",
        vae.encoder.latent_dim,
        "vae.decoder.latent_dim",
        vae.decoder.latent_dim,
    )
    _require_equal(
        "vae.encoder.latent_dim",
        vae.encoder.latent_dim,
        "reference_encoder.latent_dim",
        reference_encoder.latent_dim,
    )
    _require_equal(
        "vae.encoder.latent_dim",
        vae.encoder.latent_dim,
        "stage2_config.vae.latent_dim",
        stage2_config.vae.latent_dim,
    )
    _require_equal(
        "dit.config.latent_dim",
        dit.config.latent_dim,
        "stage2_config.dit.latent_dim",
        stage2_config.dit.latent_dim,
    )
    _require_equal(
        "stage2_config.vae.latent_dim",
        stage2_config.vae.latent_dim,
        "stage2_config.dit.latent_dim",
        stage2_config.dit.latent_dim,
    )
    _require_equal(
        "stage2_config.vae.sequence_length",
        stage2_config.vae.sequence_length,
        "stage2_config.dit.sequence_length",
        stage2_config.dit.sequence_length,
    )
    _require_equal(
        "dit.config.sequence_length",
        dit.config.sequence_length,
        "stage2_config.dit.sequence_length",
        stage2_config.dit.sequence_length,
    )
    _require_equal(
        "dit.config.block_size",
        dit.config.block_size,
        "stage2_config.dit.block_size",
        stage2_config.dit.block_size,
    )
    _require_equal(
        "vae.encoder.vocab_size",
        vae.encoder.vocab_size,
        "vae.decoder.vocab_size",
        vae.decoder.vocab_size,
    )
    _require_equal(
        "vae.encoder.vocab_size",
        vae.encoder.vocab_size,
        "reference_encoder.vocab_size",
        reference_encoder.vocab_size,
    )
    _require_equal(
        "vae.encoder.vocab_size",
        vae.encoder.vocab_size,
        "stage2_config.vae.vocab_size",
        stage2_config.vae.vocab_size,
    )
    _require_equal(
        "vae.encoder.patch_size",
        vae.encoder.patch_size,
        "vae.decoder.patch_size",
        vae.decoder.patch_size,
    )
    _require_equal(
        "vae.encoder.patch_size",
        vae.encoder.patch_size,
        "reference_encoder.patch_size",
        reference_encoder.patch_size,
    )
    _require_equal(
        "vae.encoder.patch_size",
        vae.encoder.patch_size,
        "stage2_config.vae.patch_size",
        stage2_config.vae.patch_size,
    )


def _require_instance(name: str, value: object, expected_type: type[object]) -> None:
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be a {expected_type.__name__}")


def _require_equal(
    left_name: str,
    left_value: int,
    right_name: str,
    right_value: int,
) -> None:
    if left_value != right_value:
        raise ValueError(
            f"{left_name} must match {right_name} "
            f"(got {left_value!r} and {right_value!r})"
        )


__all__ = ("create_frozen_reference_encoder",)
