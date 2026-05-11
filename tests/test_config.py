from cola_dlm.config import (
    DiTConfig,
    DiffusionConfig,
    InferenceConfig,
    OptimizerConfig,
    Stage1Config,
    Stage2Config,
    VAEConfig,
)


def test_default_configs_use_paper_scale_values():
    vae = VAEConfig()
    dit = DiTConfig()
    diffusion = DiffusionConfig()
    optimizer = OptimizerConfig()
    stage1 = Stage1Config()
    stage2 = Stage2Config()
    inference = InferenceConfig()

    assert vae.vocab_size == 100_278
    assert vae.sequence_length == 512
    assert vae.latent_dim == 16
    assert vae.encoder_layers == 4
    assert vae.decoder_layers == 4
    assert vae.hidden_size == 1_536
    assert vae.ffn_size == 6_144

    assert dit.sequence_length == 512
    assert dit.latent_dim == 16
    assert dit.block_size == 16
    assert dit.num_layers == 24
    assert dit.hidden_size == 2_048
    assert dit.ffn_size == 8_192
    assert dit.num_attention_heads == 16
    assert dit.attention_head_dim == 128

    assert diffusion.timestep_schedule == "logit_normal"
    assert diffusion.logit_normal_loc == 1.0
    assert diffusion.logit_normal_scale is None
    assert diffusion.prediction_type == "velocity"

    assert optimizer.name == "adamw"
    assert optimizer.peak_lr == 1.5e-4
    assert optimizer.betas == (0.9, 0.95)
    assert optimizer.weight_decay == 0.01
    assert optimizer.grad_clip == 1.0
    assert optimizer.warmup_steps == 5_000
    assert optimizer.warmup_start_lr == 1.0e-6
    assert optimizer.min_lr == 1.0e-5

    assert stage1.global_batch_size == 1_408
    assert stage1.tokens_per_step == 720_896
    assert stage1.kl_weight is None
    assert stage1.mask_loss_weight is None
    assert stage2.vae_dit_lr_ratio == 1.0
    assert stage2.global_batch_size == 1_408
    assert stage2.tokens_per_step == 720_896
    assert stage2.vae_loss_weight is None
    assert stage2.flow_matching_loss_weight is None
    assert stage2.reference_kl_weight is None
    assert inference.denoising_steps == 16
    assert inference.cfg_scale == 7.0
    assert inference.max_new_tokens == 32
    assert inference.condition_strategy == "clean_condition_repaint"


def test_tiny_stage_configs_retain_supplied_subconfigs(
    tiny_vae_config,
    tiny_dit_config,
    tiny_diffusion_config,
    tiny_optimizer_config,
    tiny_stage1_config,
    tiny_stage2_config,
):
    assert tiny_stage1_config.vae is tiny_vae_config
    assert tiny_stage1_config.optimizer is tiny_optimizer_config

    assert tiny_stage2_config.vae is tiny_vae_config
    assert tiny_stage2_config.dit is tiny_dit_config
    assert tiny_stage2_config.diffusion is tiny_diffusion_config
    assert tiny_stage2_config.optimizer is tiny_optimizer_config


def test_tiny_inference_config_retains_supplied_subconfigs(
    tiny_vae_config,
    tiny_dit_config,
    tiny_diffusion_config,
    tiny_inference_config,
):
    assert tiny_inference_config.vae is tiny_vae_config
    assert tiny_inference_config.dit is tiny_dit_config
    assert tiny_inference_config.diffusion is tiny_diffusion_config


def test_tiny_config_block_and_sequence_lengths_are_consistent(
    tiny_vae_config,
    tiny_dit_config,
    tiny_stage1_config,
    tiny_stage2_config,
    tiny_inference_config,
):
    assert tiny_vae_config.sequence_length == tiny_dit_config.sequence_length
    assert tiny_vae_config.latent_dim == tiny_dit_config.latent_dim
    assert tiny_vae_config.sequence_length % tiny_vae_config.patch_size == 0
    assert tiny_dit_config.sequence_length % tiny_dit_config.block_size == 0
    assert tiny_dit_config.block_size <= tiny_dit_config.sequence_length
    assert tiny_inference_config.max_new_tokens <= tiny_dit_config.block_size
    assert (
        tiny_stage1_config.tokens_per_step
        == tiny_stage1_config.global_batch_size * tiny_vae_config.sequence_length
    )
    assert (
        tiny_stage2_config.tokens_per_step
        == tiny_stage2_config.global_batch_size * tiny_dit_config.sequence_length
    )
