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


def test_tiny_configs_can_be_supplied_explicitly():
    vae = VAEConfig(
        tokenizer_name="tiny",
        vocab_size=128,
        sequence_length=16,
        latent_dim=4,
        encoder_layers=1,
        decoder_layers=1,
        hidden_size=32,
        ffn_size=64,
    )
    dit = DiTConfig(
        sequence_length=16,
        latent_dim=4,
        block_size=4,
        num_layers=2,
        hidden_size=32,
        ffn_size=64,
        num_attention_heads=4,
        attention_head_dim=8,
    )
    diffusion = DiffusionConfig(logit_normal_loc=0.0, logit_normal_scale=0.5)
    optimizer = OptimizerConfig(peak_lr=1.0e-4, warmup_steps=2)

    stage1 = Stage1Config(
        vae=vae,
        optimizer=optimizer,
        global_batch_size=2,
        tokens_per_step=32,
        kl_weight=0.1,
        mask_loss_weight=0.2,
    )
    stage2 = Stage2Config(
        vae=vae,
        dit=dit,
        diffusion=diffusion,
        optimizer=optimizer,
        global_batch_size=2,
        tokens_per_step=32,
        vae_loss_weight=0.3,
        flow_matching_loss_weight=0.4,
        reference_kl_weight=0.5,
    )
    inference = InferenceConfig(
        vae=vae,
        dit=dit,
        diffusion=diffusion,
        denoising_steps=2,
        cfg_scale=1.0,
        max_new_tokens=4,
    )

    assert stage1.vae is vae
    assert stage2.vae is vae
    assert stage2.dit is dit
    assert stage2.diffusion is diffusion
    assert stage2.optimizer is optimizer
    assert inference.vae is vae
    assert inference.dit is dit
    assert inference.diffusion is diffusion
