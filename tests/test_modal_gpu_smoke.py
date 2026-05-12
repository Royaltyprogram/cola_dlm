import math

import pytest

from cola_dlm.modal_gpu_smoke import (
    build_tiny_modal_smoke_config,
    run_tiny_stage2_smoke_step,
)


def test_build_tiny_modal_smoke_config_uses_expected_small_dimensions():
    config = build_tiny_modal_smoke_config()

    assert config.vae.sequence_length == 8
    assert config.dit.sequence_length == 8
    assert config.dit.block_size == 2
    assert config.vae.vocab_size == 32
    assert config.vae.latent_dim == 2
    assert config.dit.latent_dim == 2
    assert config.vae.encoder_layers == 1
    assert config.vae.decoder_layers == 1
    assert config.dit.num_layers == 1


def test_tiny_stage2_smoke_step_runs_one_cpu_step():
    result = run_tiny_stage2_smoke_step()

    assert result["success"] is True
    assert result["device"] == "cpu"
    assert isinstance(result["cuda_available"], bool)
    assert isinstance(result["loss"], float)
    assert math.isfinite(result["loss"])
    assert result["steps"] == 1


def test_tiny_stage2_smoke_step_reports_cpu_placement():
    result = run_tiny_stage2_smoke_step()

    assert result["vae_parameter_device"] == "cpu"
    assert result["dit_parameter_device"] == "cpu"
    assert result["token_device"] == "cpu"


def test_tiny_stage2_smoke_step_requires_cuda_when_requested():
    with pytest.raises(RuntimeError, match="CUDA is required"):
        run_tiny_stage2_smoke_step(device="cpu", require_cuda=True)
