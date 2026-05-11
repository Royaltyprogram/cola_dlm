from dataclasses import FrozenInstanceError, fields, replace

import pytest
import torch

from cola_dlm.config import InferenceConfig
from cola_dlm.dit import BlockCausalTextDiT
from cola_dlm.inference import (
    GenerationBlock,
    InferenceOutput,
    apply_clean_condition_repaint,
    combine_cfg_vector_fields,
    encode_prefix_latents,
    generate,
    iter_generation_blocks,
    sample_latent_blocks,
)
from cola_dlm.vae import TextVAE


def test_inference_public_surface():
    import cola_dlm.inference as inference

    assert inference.__all__ == (
        "GenerationBlock",
        "InferenceOutput",
        "apply_clean_condition_repaint",
        "combine_cfg_vector_fields",
        "encode_prefix_latents",
        "generate",
        "iter_generation_blocks",
        "sample_latent_blocks",
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


def test_first_mixed_generation_block_preserves_known_prefix_latents(
    tiny_inference_config,
):
    prefix_latents = torch.arange(
        1 * 6 * tiny_inference_config.dit.latent_dim,
        dtype=torch.float32,
    ).view(1, 6, tiny_inference_config.dit.latent_dim)
    blocks = list(
        iter_generation_blocks(
            prefix_length=prefix_latents.shape[1],
            num_new_latents=3,
            block_size=4,
        )
    )
    first_block = blocks[0]

    assert (first_block.start, first_block.end) == (4, 8)
    assert first_block.known_mask.tolist() == [True, True, False, False]

    block_latents = torch.full(
        (1, 4, tiny_inference_config.dit.latent_dim),
        -1.0,
    )
    clean_block_latents = torch.zeros_like(block_latents)
    clean_block_latents[:, :2] = prefix_latents[:, 4:6]
    repainted = apply_clean_condition_repaint(
        block_latents,
        clean_block_latents,
        first_block.known_mask,
    )

    torch.testing.assert_close(repainted[:, :2], prefix_latents[:, 4:6])
    torch.testing.assert_close(repainted[:, 2:], block_latents[:, 2:])

    dit = _ConstantDiT(tiny_inference_config.dit, value=0.0)
    config = replace(tiny_inference_config, num_denoise_steps=1, cfg_scale=1.0)
    generated = sample_latent_blocks(
        dit,
        prefix_latents,
        inference_config=config,
        num_new_latents=3,
        block_size=4,
        generator=torch.Generator().manual_seed(59),
    )

    assert generated.shape == (1, 3, tiny_inference_config.dit.latent_dim)
    first_packed_latents = dit.packed_latent_calls[0]
    torch.testing.assert_close(
        first_packed_latents[:, 4:6],
        prefix_latents[:, 4:6],
    )
    assert torch.isfinite(first_packed_latents[:, 6:8]).all()


def test_denoise_inference_block_repaints_known_positions_exactly(
    tiny_inference_config,
):
    import cola_dlm.inference as inference

    dit = _ConstantDiT(tiny_inference_config.dit, value=1.0)
    config = replace(tiny_inference_config, num_denoise_steps=2, cfg_scale=1.0)
    prefix_latents = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
    clean_history = prefix_latents[:, :0]
    block_latents = torch.randn(
        1,
        4,
        4,
        generator=torch.Generator().manual_seed(17),
    )
    initial_unknown = block_latents[:, 3:].clone()
    clean_block_latents = torch.zeros_like(block_latents)
    clean_block_latents[:, :3] = prefix_latents
    known_mask = torch.tensor([True, True, True, False])

    denoised = inference._denoise_inference_block(
        dit=dit,
        clean_history_latents=clean_history,
        block_latents=block_latents,
        block_start=0,
        known_mask=known_mask,
        clean_block_latents=clean_block_latents,
        config=config,
        cfg_scale=1.0,
        block_size=4,
    )

    torch.testing.assert_close(denoised[:, :3], prefix_latents)
    torch.testing.assert_close(denoised[:, 3:], initial_unknown - 1.0)


def test_pack_inference_block_uses_absolute_blocks_and_segments():
    import cola_dlm.inference as inference

    clean_history = torch.zeros(1, 6, 2)
    block_latents = torch.ones(1, 2, 2)

    packed = inference._pack_inference_block(
        clean_history,
        block_latents,
        block_start=6,
        block_size=2,
    )

    assert packed.block_ids.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert packed.segment_ids.tolist() == [0, 0, 0, 0, 0, 0, 1, 1]
    assert packed.current_slice == slice(6, 8)
    torch.testing.assert_close(packed.latents[:, :6], clean_history)
    torch.testing.assert_close(packed.latents[:, 6:], block_latents)


def test_initial_block_latents_samples_only_unknown_positions():
    import cola_dlm.inference as inference

    prefix_latents = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
    block = GenerationBlock(
        start=0,
        end=4,
        known_mask=torch.tensor([True, True, True, False]),
    )

    latents = inference._initial_block_latents(
        prefix_latents,
        block,
        block.known_mask,
        generator=torch.Generator().manual_seed(41),
    )
    expected_unknown = torch.randn(
        1,
        1,
        4,
        generator=torch.Generator().manual_seed(41),
    )

    torch.testing.assert_close(latents[:, :3], prefix_latents)
    torch.testing.assert_close(latents[:, 3:], expected_unknown)


def test_sample_latent_blocks_num_denoise_steps_controls_update_count(
    tiny_inference_config,
):
    dit = _ConstantDiT(tiny_inference_config.dit, value=1.0)
    config = replace(tiny_inference_config, num_denoise_steps=3, cfg_scale=1.0)
    prefix_latents = torch.zeros(1, 4, tiny_inference_config.dit.latent_dim)

    generated = sample_latent_blocks(
        dit,
        prefix_latents,
        inference_config=config,
        num_new_latents=2,
        generator=torch.Generator().manual_seed(23),
    )

    assert generated.shape == (1, 2, tiny_inference_config.dit.latent_dim)
    assert dit.call_count == 3


def test_sample_latent_blocks_block_size_controls_generation_blocks(
    tiny_inference_config,
):
    prefix_latents = torch.zeros(1, 1, tiny_inference_config.dit.latent_dim)
    config = replace(tiny_inference_config, num_denoise_steps=1, cfg_scale=1.0)

    dit_for_size_four = _ConstantDiT(tiny_inference_config.dit, value=0.0)
    sample_latent_blocks(
        dit_for_size_four,
        prefix_latents,
        inference_config=config,
        num_new_latents=7,
        block_size=4,
        generator=torch.Generator().manual_seed(29),
    )

    dit_for_size_two = _ConstantDiT(tiny_inference_config.dit, value=0.0)
    sample_latent_blocks(
        dit_for_size_two,
        prefix_latents,
        inference_config=config,
        num_new_latents=7,
        block_size=2,
        generator=torch.Generator().manual_seed(29),
    )

    assert dit_for_size_four.call_count == 2
    assert dit_for_size_two.call_count == 4


def test_sample_latent_blocks_cfg_scale_one_skips_unconditional_branch(
    tiny_inference_config,
):
    dit = _ConstantDiT(tiny_inference_config.dit, value=2.0)
    config = replace(tiny_inference_config, num_denoise_steps=2, cfg_scale=1.0)
    prefix_latents = torch.zeros(1, 4, tiny_inference_config.dit.latent_dim)
    unconditional_prefix_latents = prefix_latents + 10.0

    generated = sample_latent_blocks(
        dit,
        prefix_latents,
        inference_config=config,
        num_new_latents=1,
        unconditional_prefix_latents=unconditional_prefix_latents,
        generator=torch.Generator().manual_seed(31),
    )
    expected_noise = torch.randn(
        1,
        1,
        tiny_inference_config.dit.latent_dim,
        generator=torch.Generator().manual_seed(31),
    )

    assert dit.call_count == 2
    torch.testing.assert_close(generated, expected_noise - 2.0)


@pytest.mark.parametrize(
    ("cfg_scale", "expected"),
    [
        (0.0, 1.0),
        (1.0, 3.0),
        (2.5, 6.0),
    ],
)
def test_combine_cfg_vector_fields_uses_expected_linear_combination(
    cfg_scale,
    expected,
):
    unconditional = torch.ones(2, 3, 4)
    conditional = unconditional + 2.0

    guided = combine_cfg_vector_fields(unconditional, conditional, cfg_scale)

    torch.testing.assert_close(guided, torch.full_like(guided, expected))


@pytest.mark.parametrize("sampler", ["euler", "heun"])
def test_sample_latent_blocks_euler_and_heun_return_finite_tiny_latents(
    tiny_inference_config,
    sampler,
):
    torch.manual_seed(0)
    dit = BlockCausalTextDiT(tiny_inference_config.dit)
    dit.eval()
    config = replace(
        tiny_inference_config,
        sampler=sampler,
        num_denoise_steps=1,
        cfg_scale=1.0,
    )
    prefix_latents = torch.randn(2, 3, tiny_inference_config.dit.latent_dim)

    with torch.no_grad():
        generated = sample_latent_blocks(
            dit,
            prefix_latents,
            inference_config=config,
            num_new_latents=2,
            generator=torch.Generator().manual_seed(37),
        )

    assert generated.shape == (2, 2, tiny_inference_config.dit.latent_dim)
    assert torch.isfinite(generated).all()


def test_generate_returns_decoded_response_shapes(tiny_inference_config):
    torch.manual_seed(0)
    vae = TextVAE(tiny_inference_config.vae)
    dit = BlockCausalTextDiT(tiny_inference_config.dit)
    vae.eval()
    dit.eval()
    prefix_token_ids = torch.randint(tiny_inference_config.vae.vocab_size, (2, 5))

    with torch.no_grad():
        output = generate(
            vae,
            dit,
            prefix_token_ids,
            inference_config=tiny_inference_config,
            max_new_tokens=3,
            generator=torch.Generator().manual_seed(43),
        )

    assert output.generated_latents.shape == (
        2,
        3,
        tiny_inference_config.vae.latent_dim,
    )
    assert output.all_latents.shape[1] == prefix_token_ids.shape[1] + 3
    assert output.response_logits.shape == (
        2,
        3,
        tiny_inference_config.vae.vocab_size,
    )
    assert output.response_token_ids.shape == (2, 3)
    assert output.kv_cache is None


def test_generate_call_time_max_new_tokens_overrides_config(
    tiny_inference_config,
):
    torch.manual_seed(0)
    vae = TextVAE(tiny_inference_config.vae)
    dit = BlockCausalTextDiT(tiny_inference_config.dit)
    vae.eval()
    dit.eval()
    config = replace(
        tiny_inference_config,
        max_new_tokens=1,
        num_denoise_steps=1,
    )
    prefix_token_ids = torch.randint(config.vae.vocab_size, (1, 4))

    with torch.no_grad():
        output = generate(
            vae,
            dit,
            prefix_token_ids,
            inference_config=config,
            max_new_tokens=3,
            generator=torch.Generator().manual_seed(47),
        )

    assert output.generated_latents.shape[1] == 3
    assert output.response_logits.shape[1] == 3
    assert output.response_token_ids.shape == (1, 3)


def test_generate_rejects_kv_cache(tiny_inference_config):
    vae = TextVAE(tiny_inference_config.vae)
    dit = BlockCausalTextDiT(tiny_inference_config.dit)
    prefix_token_ids = torch.randint(tiny_inference_config.vae.vocab_size, (1, 4))

    with pytest.raises(ValueError, match="kv_cache is not supported yet"):
        generate(
            vae,
            dit,
            prefix_token_ids,
            inference_config=tiny_inference_config,
            kv_cache=object(),
        )


def test_generate_rejects_unsupported_condition_strategy(tiny_inference_config):
    vae = TextVAE(tiny_inference_config.vae)
    dit = BlockCausalTextDiT(tiny_inference_config.dit)
    config = replace(tiny_inference_config, condition_strategy="left_pad")
    prefix_token_ids = torch.randint(config.vae.vocab_size, (1, 4))

    with pytest.raises(ValueError, match="only supports 'clean_condition_repaint'"):
        generate(
            vae,
            dit,
            prefix_token_ids,
            inference_config=config,
            max_new_tokens=1,
        )


def test_generate_preserves_unaligned_prefix_latents(tiny_inference_config):
    torch.manual_seed(0)
    vae = TextVAE(tiny_inference_config.vae)
    dit = BlockCausalTextDiT(tiny_inference_config.dit)
    vae.eval()
    dit.eval()
    config = replace(
        tiny_inference_config,
        max_new_tokens=2,
        num_denoise_steps=1,
    )
    prefix_token_ids = torch.randint(config.vae.vocab_size, (1, 3))

    with torch.no_grad():
        output = generate(
            vae,
            dit,
            prefix_token_ids,
            inference_config=config,
            generator=torch.Generator().manual_seed(53),
        )

    torch.testing.assert_close(
        output.all_latents[:, : prefix_token_ids.shape[1], :],
        output.prefix_latents,
    )
    assert output.generated_latents.shape[1] == 2


class _ConstantDiT:
    def __init__(self, config, *, value: float) -> None:
        self.config = config
        self.value = value
        self.call_count = 0
        self.packed_latent_calls = []

    def __call__(
        self,
        packed_latents,
        timesteps,
        attention_mask,
        segment_ids,
    ):
        self.call_count += 1
        assert timesteps.shape == (packed_latents.shape[0],)
        assert attention_mask.shape == (
            packed_latents.shape[1],
            packed_latents.shape[1],
        )
        assert segment_ids.shape == (packed_latents.shape[1],)
        self.packed_latent_calls.append(packed_latents.detach().clone())
        return torch.full_like(packed_latents, self.value)
