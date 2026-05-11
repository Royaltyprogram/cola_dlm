import pytest

from cola_dlm.config import (
    DiTConfig,
    DiffusionConfig,
    InferenceConfig,
    OptimizerConfig,
    Stage1Config,
    Stage2Config,
    VAEConfig,
)


@pytest.fixture
def tiny_vae_config() -> VAEConfig:
    return VAEConfig(
        tokenizer_name="tiny",
        vocab_size=128,
        sequence_length=16,
        latent_dim=4,
        patch_size=1,
        encoder_layers=1,
        decoder_layers=1,
        hidden_size=32,
        ffn_size=64,
    )


@pytest.fixture
def tiny_dit_config() -> DiTConfig:
    return DiTConfig(
        sequence_length=16,
        latent_dim=4,
        block_size=4,
        num_layers=2,
        hidden_size=32,
        ffn_size=64,
        num_attention_heads=4,
        attention_head_dim=8,
    )


@pytest.fixture
def tiny_diffusion_config() -> DiffusionConfig:
    return DiffusionConfig(logit_normal_loc=0.0, logit_normal_scale=0.5)


@pytest.fixture
def tiny_optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(peak_lr=1.0e-4, warmup_steps=2)


@pytest.fixture
def tiny_stage1_config(
    tiny_vae_config: VAEConfig,
    tiny_optimizer_config: OptimizerConfig,
) -> Stage1Config:
    return Stage1Config(
        vae=tiny_vae_config,
        optimizer=tiny_optimizer_config,
        global_batch_size=2,
        tokens_per_step=32,
        kl_weight=0.1,
        mask_loss_weight=0.2,
    )


@pytest.fixture
def tiny_stage2_config(
    tiny_vae_config: VAEConfig,
    tiny_dit_config: DiTConfig,
    tiny_diffusion_config: DiffusionConfig,
    tiny_optimizer_config: OptimizerConfig,
) -> Stage2Config:
    return Stage2Config(
        vae=tiny_vae_config,
        dit=tiny_dit_config,
        diffusion=tiny_diffusion_config,
        optimizer=tiny_optimizer_config,
        global_batch_size=2,
        tokens_per_step=32,
        vae_loss_weight=0.3,
        flow_matching_loss_weight=0.4,
        reference_kl_weight=0.5,
    )


@pytest.fixture
def tiny_inference_config(
    tiny_vae_config: VAEConfig,
    tiny_dit_config: DiTConfig,
    tiny_diffusion_config: DiffusionConfig,
) -> InferenceConfig:
    return InferenceConfig(
        vae=tiny_vae_config,
        dit=tiny_dit_config,
        diffusion=tiny_diffusion_config,
        denoising_steps=2,
        cfg_scale=1.0,
        max_new_tokens=4,
    )
