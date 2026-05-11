from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from cola_dlm.config import InferenceConfig
from cola_dlm.inference import (
    GenerationBlock,
    InferenceOutput,
    apply_clean_condition_repaint,
    encode_prefix_latents,
    iter_generation_blocks,
)
from cola_dlm.vae import TextVAE


def test_inference_public_surface():
    import cola_dlm.inference as inference

    assert inference.__all__ == (
        "GenerationBlock",
        "InferenceOutput",
        "apply_clean_condition_repaint",
        "encode_prefix_latents",
        "iter_generation_blocks",
    )


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


def test_encode_prefix_latents_returns_per_token_latents(tiny_vae_config):
    torch.manual_seed(0)
    vae = TextVAE(config=tiny_vae_config)
    prefix_token_ids = torch.randint(tiny_vae_config.vocab_size, (2, 5))
    attention_mask = torch.ones_like(prefix_token_ids, dtype=torch.bool)

    prefix_latents = encode_prefix_latents(
        vae,
        prefix_token_ids,
        attention_mask=attention_mask,
    )

    assert prefix_latents.shape == (2, 5, tiny_vae_config.latent_dim)


def test_encode_prefix_latents_deterministic_mode_matches_encoder_mode(
    tiny_vae_config,
):
    torch.manual_seed(0)
    vae = TextVAE(config=tiny_vae_config)
    vae.eval()
    prefix_token_ids = torch.randint(tiny_vae_config.vocab_size, (2, 5))

    with torch.no_grad():
        expected = vae.encoder(prefix_token_ids).mode()
        prefix_latents = encode_prefix_latents(vae, prefix_token_ids)

    assert torch.equal(prefix_latents, expected)


def test_encode_prefix_latents_validates_token_ids(tiny_vae_config):
    vae = TextVAE(config=tiny_vae_config)

    with pytest.raises(ValueError, match="prefix_token_ids must be a torch.long"):
        encode_prefix_latents(vae, torch.zeros(2, 4))

    invalid_shape = torch.zeros(2, 4, 1, dtype=torch.long)
    with pytest.raises(ValueError, match=r"prefix_token_ids must be shaped"):
        encode_prefix_latents(vae, invalid_shape)


def test_encode_prefix_latents_validates_attention_mask(tiny_vae_config):
    vae = TextVAE(config=tiny_vae_config)
    prefix_token_ids = torch.randint(tiny_vae_config.vocab_size, (2, 4))

    invalid_dtype = torch.ones(2, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="attention_mask must be a boolean tensor"):
        encode_prefix_latents(
            vae,
            prefix_token_ids,
            attention_mask=invalid_dtype,
        )

    invalid_shape = torch.ones(2, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="attention_mask must match"):
        encode_prefix_latents(
            vae,
            prefix_token_ids,
            attention_mask=invalid_shape,
        )


def test_iter_generation_blocks_handles_aligned_prefixes():
    blocks = list(
        iter_generation_blocks(
            prefix_length=4,
            num_new_latents=6,
            block_size=4,
        )
    )

    assert [(block.start, block.end) for block in blocks] == [(4, 8), (8, 10)]
    assert all(isinstance(block, GenerationBlock) for block in blocks)
    start, end, known_mask = blocks[0]
    assert (start, end, known_mask.tolist()) == (
        4,
        8,
        [False, False, False, False],
    )
    assert [block.known_mask.tolist() for block in blocks] == [
        [False, False, False, False],
        [False, False],
    ]


def test_iter_generation_blocks_marks_mixed_first_block_prefix_positions():
    blocks = list(
        iter_generation_blocks(
            prefix_length=6,
            num_new_latents=5,
            block_size=4,
        )
    )

    assert [(block.start, block.end) for block in blocks] == [(4, 8), (8, 11)]
    assert [block.known_mask.tolist() for block in blocks] == [
        [True, True, False, False],
        [False, False, False],
    ]


def test_iter_generation_blocks_handles_final_partial_generated_block():
    blocks = list(
        iter_generation_blocks(
            prefix_length=1,
            num_new_latents=10,
            block_size=4,
        )
    )

    assert [(block.start, block.end) for block in blocks] == [
        (0, 4),
        (4, 8),
        (8, 11),
    ]
    assert blocks[0].known_mask.tolist() == [True, False, False, False]
    assert blocks[-1].known_mask.tolist() == [False, False, False]


def test_iter_generation_blocks_rejects_unsupported_condition_strategy():
    with pytest.raises(ValueError, match="only supports 'clean_condition_repaint'"):
        list(
            iter_generation_blocks(
                prefix_length=3,
                num_new_latents=2,
                block_size=4,
                condition_strategy="left_pad",
            )
        )


def test_apply_clean_condition_repaint_keeps_unknown_positions_unchanged():
    block_latents = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    clean_block_latents = block_latents + 100.0
    known_mask = torch.tensor([True, False, True])

    repainted = apply_clean_condition_repaint(
        block_latents,
        clean_block_latents,
        known_mask,
    )

    assert torch.equal(
        repainted[:, known_mask, :],
        clean_block_latents[:, known_mask, :],
    )
    assert torch.equal(
        repainted[:, ~known_mask, :],
        block_latents[:, ~known_mask, :],
    )
    assert torch.equal(
        block_latents,
        torch.arange(24, dtype=torch.float32).view(2, 3, 4),
    )


def test_apply_clean_condition_repaint_accepts_batched_known_masks():
    block_latents = torch.zeros(2, 3, 2)
    clean_block_latents = torch.ones_like(block_latents)
    known_mask = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
        ]
    )

    repainted = apply_clean_condition_repaint(
        block_latents,
        clean_block_latents,
        known_mask,
    )

    assert torch.equal(repainted[0, 0], clean_block_latents[0, 0])
    assert torch.equal(repainted[1, 1], clean_block_latents[1, 1])
    assert torch.equal(repainted[0, 1:], block_latents[0, 1:])
    assert torch.equal(repainted[1, [0, 2]], block_latents[1, [0, 2]])


def test_apply_clean_condition_repaint_repeatedly_restores_known_positions():
    block_latents = torch.randn(2, 4, 3)
    clean_block_latents = torch.randn(2, 4, 3)
    known_mask = torch.tensor([False, True, True, False])

    repainted = apply_clean_condition_repaint(
        block_latents,
        clean_block_latents,
        known_mask,
    )
    drifted = repainted.clone()
    drifted[:, known_mask, :] += 10.0

    restored = apply_clean_condition_repaint(
        drifted,
        clean_block_latents,
        known_mask,
    )

    assert torch.equal(
        restored[:, known_mask, :],
        clean_block_latents[:, known_mask, :],
    )
    assert torch.equal(restored[:, ~known_mask, :], drifted[:, ~known_mask, :])
