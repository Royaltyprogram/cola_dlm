from dataclasses import FrozenInstanceError, fields

import pytest

from cola_dlm.config import InferenceConfig
from cola_dlm.inference import InferenceOutput


def test_inference_public_surface():
    import cola_dlm.inference as inference

    assert inference.__all__ == ("InferenceOutput",)


def test_default_inference_config_values():
    config = InferenceConfig()

    assert config.num_denoise_steps == 16
    assert config.sampler == "euler"
    assert config.cfg_scale == 7.0
    assert config.max_new_tokens == 32


def test_inference_config_rejects_invalid_num_denoise_steps():
    with pytest.raises(ValueError, match="num_denoise_steps must be positive"):
        InferenceConfig(num_denoise_steps=0)


def test_inference_config_rejects_invalid_sampler():
    with pytest.raises(ValueError, match="sampler must be 'euler' or 'heun'"):
        InferenceConfig(sampler="ddim")


def test_inference_config_rejects_negative_cfg_scale():
    with pytest.raises(ValueError, match="cfg_scale must be non-negative"):
        InferenceConfig(cfg_scale=-0.1)


def test_inference_output_is_frozen_and_preserves_field_names():
    output = InferenceOutput(
        prefix_latents="prefix",
        generated_latents="generated",
        all_latents="all",
        response_logits="logits",
        response_token_ids="tokens",
        kv_cache=None,
    )

    assert tuple(field.name for field in fields(InferenceOutput)) == (
        "prefix_latents",
        "generated_latents",
        "all_latents",
        "response_logits",
        "response_token_ids",
        "kv_cache",
    )
    assert output.kv_cache is None
    with pytest.raises(FrozenInstanceError):
        output.response_logits = "changed"


def test_inference_output_rejects_kv_cache_placeholder_values():
    with pytest.raises(ValueError, match="kv_cache is not supported yet"):
        InferenceOutput(
            prefix_latents=None,
            generated_latents=None,
            all_latents=None,
            response_logits=None,
            response_token_ids=None,
            kv_cache=object(),
        )
