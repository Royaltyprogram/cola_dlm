from dataclasses import replace

import pytest
import torch

from cola_dlm.block_causal_mask import build_packed_dit_inputs
from cola_dlm.dit import BlockCausalTextDiT, TimestepEmbedding


def test_timestep_embedding_accepts_vector_and_column_timesteps():
    torch.manual_seed(0)
    embedding = TimestepEmbedding(hidden_size=8)
    timesteps = torch.tensor([0.0, 0.5, 1.0])

    vector_output = embedding(timesteps)
    column_output = embedding(timesteps[:, None])

    assert vector_output.shape == (3, 8)
    assert column_output.shape == (3, 8)
    torch.testing.assert_close(vector_output, column_output)


@pytest.mark.parametrize(
    "timesteps",
    [
        torch.tensor(0.5),
        torch.ones(2, 2),
        torch.ones(2, 1, 1),
    ],
)
def test_timestep_embedding_rejects_invalid_timestep_rank_or_shape(timesteps):
    embedding = TimestepEmbedding(hidden_size=8)

    with pytest.raises(ValueError, match=r"timesteps must be shaped \[batch\]"):
        embedding(timesteps)


def test_timestep_embedding_rejects_non_floating_timesteps():
    embedding = TimestepEmbedding(hidden_size=8)
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="floating point"):
        embedding(timesteps)


def test_block_causal_text_dit_predicts_packed_latent_shape(tiny_dit_config):
    torch.manual_seed(0)
    model = BlockCausalTextDiT(tiny_dit_config).to(dtype=torch.float64)
    model.eval()
    z0, zt = _tiny_stage2_latents(tiny_dit_config, dtype=torch.float64)
    packed = build_packed_dit_inputs(z0, zt, block_size=tiny_dit_config.block_size)
    timesteps = torch.tensor([0.25, 0.75], dtype=torch.float64)

    with torch.no_grad():
        output = model(packed.latents, timesteps, packed.attention_mask)

    assert output.shape == packed.latents.shape
    assert output.dtype is torch.float64
    assert torch.isfinite(output).all()


def test_block_causal_text_dit_accepts_packed_segment_ids(tiny_dit_config):
    torch.manual_seed(0)
    config = replace(tiny_dit_config, use_segment_embedding=True)
    model = BlockCausalTextDiT(config)
    model.eval()
    z0, zt = _tiny_stage2_latents(config)
    packed = build_packed_dit_inputs(z0, zt, block_size=config.block_size)
    timesteps = torch.tensor([0.25, 0.75])

    with torch.no_grad():
        output = model(
            packed.latents,
            timesteps,
            packed.attention_mask,
            segment_ids=packed.segment_ids,
        )

    assert output.shape == packed.latents.shape
    assert torch.isfinite(output).all()


def test_block_causal_text_dit_loss_mask_selects_noisy_targets(tiny_dit_config):
    torch.manual_seed(0)
    model = BlockCausalTextDiT(tiny_dit_config)
    model.eval()
    z0, zt = _tiny_stage2_latents(tiny_dit_config)
    packed = build_packed_dit_inputs(z0, zt, block_size=tiny_dit_config.block_size)
    timesteps = torch.tensor([0.25, 0.75])

    with torch.no_grad():
        predictions = model(packed.latents, timesteps, packed.attention_mask)

    masked_predictions = predictions[packed.loss_mask]
    expected_noisy_targets = predictions[:, -tiny_dit_config.sequence_length :].reshape(
        -1,
        tiny_dit_config.latent_dim,
    )

    assert masked_predictions.shape == (
        packed.latents.shape[0] * tiny_dit_config.sequence_length,
        tiny_dit_config.latent_dim,
    )
    torch.testing.assert_close(masked_predictions, expected_noisy_targets)


def test_block_causal_text_dit_allows_missing_segments_when_disabled(
    tiny_dit_config,
):
    model = BlockCausalTextDiT(replace(tiny_dit_config, use_segment_embedding=False))
    model.eval()
    z0, zt = _tiny_stage2_latents(tiny_dit_config)
    packed = build_packed_dit_inputs(z0, zt, block_size=tiny_dit_config.block_size)
    timesteps = torch.tensor([0.25, 0.75])

    with torch.no_grad():
        output = model(
            packed.latents,
            timesteps,
            packed.attention_mask,
            segment_ids=None,
        )

    assert output.shape == packed.latents.shape


def test_block_causal_text_dit_rejects_missing_segments_when_enabled(
    tiny_dit_config,
):
    config = replace(tiny_dit_config, use_segment_embedding=True)
    model = BlockCausalTextDiT(config)
    z0, zt = _tiny_stage2_latents(config)
    packed = build_packed_dit_inputs(z0, zt, block_size=config.block_size)
    timesteps = torch.tensor([0.25, 0.75])

    with pytest.raises(ValueError, match="segment_ids must be provided"):
        model(packed.latents, timesteps, packed.attention_mask)


def test_block_causal_text_dit_rejects_wrongly_shaped_segments(
    tiny_dit_config,
):
    config = replace(tiny_dit_config, use_segment_embedding=True)
    model = BlockCausalTextDiT(config)
    z0, zt = _tiny_stage2_latents(config)
    packed = build_packed_dit_inputs(z0, zt, block_size=config.block_size)
    timesteps = torch.tensor([0.25, 0.75])
    invalid_segment_ids = packed.segment_ids[:-1]

    with pytest.raises(ValueError, match="segment_ids.*packed_len"):
        model(
            packed.latents,
            timesteps,
            packed.attention_mask,
            segment_ids=invalid_segment_ids,
        )


def test_block_causal_text_dit_accepts_column_timesteps(tiny_dit_config):
    torch.manual_seed(0)
    model = BlockCausalTextDiT(tiny_dit_config)
    model.eval()
    z0, zt = _tiny_stage2_latents(tiny_dit_config)
    packed = build_packed_dit_inputs(z0, zt, block_size=tiny_dit_config.block_size)
    timesteps = torch.tensor([0.25, 0.75])

    with torch.no_grad():
        vector_output = model(packed.latents, timesteps, packed.attention_mask)
        column_output = model(
            packed.latents,
            timesteps[:, None],
            packed.attention_mask,
        )

    torch.testing.assert_close(vector_output, column_output)


def test_block_causal_text_dit_rejects_mismatched_latent_dim(tiny_dit_config):
    model = BlockCausalTextDiT(tiny_dit_config)
    z0, zt = _tiny_stage2_latents(tiny_dit_config)
    packed = build_packed_dit_inputs(z0, zt, block_size=tiny_dit_config.block_size)
    invalid_latents = torch.randn(
        packed.latents.shape[0],
        packed.latents.shape[1],
        tiny_dit_config.latent_dim + 1,
    )
    timesteps = torch.tensor([0.25, 0.75])

    with pytest.raises(ValueError, match="latent dimension"):
        model(invalid_latents, timesteps, packed.attention_mask)


def test_block_causal_text_dit_rejects_mismatched_attention_mask_length(
    tiny_dit_config,
):
    model = BlockCausalTextDiT(tiny_dit_config)
    z0, zt = _tiny_stage2_latents(tiny_dit_config)
    packed = build_packed_dit_inputs(z0, zt, block_size=tiny_dit_config.block_size)
    invalid_attention_mask = packed.attention_mask[:-1, :-1]
    timesteps = torch.tensor([0.25, 0.75])

    with pytest.raises(ValueError, match="attention_mask.*packed_len"):
        model(packed.latents, timesteps, invalid_attention_mask)


def _tiny_stage2_latents(tiny_dit_config, dtype=torch.float32):
    shape = (2, tiny_dit_config.sequence_length, tiny_dit_config.latent_dim)
    return torch.randn(shape, dtype=dtype), torch.randn(shape, dtype=dtype)
