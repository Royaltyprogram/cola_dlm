import math

import pytest
import torch

from cola_dlm.config import VAEConfig
from cola_dlm.vae import (
    DiagonalGaussianPosterior,
    TextVAE,
    TextVAEDecoder,
    TextVAEEncoder,
    vae_logsnr,
)


def test_vae_public_exports_are_core_utilities():
    import cola_dlm.vae as vae

    assert vae.__all__ == (
        "DiagonalGaussianPosterior",
        "TextVAEEncoder",
        "TextVAEDecoder",
        "TextVAEOutput",
        "TextVAE",
        "vae_logsnr",
    )


def test_posterior_rejects_mismatched_shapes():
    mu = torch.zeros(2, 3, 4)
    logvar = torch.zeros(2, 3, 5)

    with pytest.raises(ValueError, match="same shape"):
        DiagonalGaussianPosterior(mu=mu, logvar=logvar)


def test_posterior_rejects_non_floating_tensors():
    mu = torch.zeros(2, 3, 4, dtype=torch.long)
    logvar = torch.zeros(2, 3, 4)

    with pytest.raises(ValueError, match="floating point"):
        DiagonalGaussianPosterior(mu=mu, logvar=logvar)


def test_posterior_sampling_preserves_shape_and_dtype():
    generator = torch.Generator().manual_seed(0)
    mu = torch.zeros(2, 3, 4)
    logvar = torch.zeros_like(mu)
    posterior = DiagonalGaussianPosterior(mu=mu, logvar=logvar)

    sample = posterior.sample(generator=generator)

    assert sample.shape == mu.shape
    assert sample.dtype == mu.dtype


def test_posterior_mode_returns_mu():
    mu = torch.randn(2, 3, 4)
    logvar = torch.zeros_like(mu)
    posterior = DiagonalGaussianPosterior(mu=mu, logvar=logvar)

    assert posterior.mode() is mu


def test_posterior_kl_to_standard_normal_is_per_token():
    mu = torch.tensor([[[0.0, 1.0], [2.0, 0.0]]])
    logvar = torch.zeros_like(mu)
    posterior = DiagonalGaussianPosterior(mu=mu, logvar=logvar)

    kl = posterior.kl()

    expected = torch.tensor([[0.5, 2.0]])
    assert kl.shape == (1, 2)
    assert torch.allclose(kl, expected)


def test_posterior_entropy_is_per_token():
    mu = torch.zeros(2, 3, 4)
    logvar = torch.zeros_like(mu)
    posterior = DiagonalGaussianPosterior(mu=mu, logvar=logvar)

    entropy = posterior.entropy()

    expected_value = 0.5 * mu.shape[-1] * (1.0 + math.log(2.0 * math.pi))
    expected = torch.full((2, 3), expected_value)
    assert entropy.shape == (2, 3)
    assert torch.allclose(entropy, expected)


def test_posterior_log_prob_is_per_token():
    mu = torch.zeros(2, 3, 4)
    logvar = torch.zeros_like(mu)
    value = torch.zeros_like(mu)
    posterior = DiagonalGaussianPosterior(mu=mu, logvar=logvar)

    log_prob = posterior.log_prob(value)

    expected_value = -0.5 * mu.shape[-1] * math.log(2.0 * math.pi)
    expected = torch.full((2, 3), expected_value)
    assert log_prob.shape == (2, 3)
    assert torch.allclose(log_prob, expected)


def test_posterior_log_prob_rejects_mismatched_value_shape():
    mu = torch.zeros(2, 3, 4)
    logvar = torch.zeros_like(mu)
    posterior = DiagonalGaussianPosterior(mu=mu, logvar=logvar)

    with pytest.raises(ValueError, match="same shape"):
        posterior.log_prob(torch.zeros(2, 3, 5))


def test_vae_logsnr_is_finite_for_zero_signal():
    mu = torch.zeros(2, 3, 4)
    logvar = torch.zeros_like(mu)

    logsnr = vae_logsnr(mu, logvar)

    assert logsnr.shape == ()
    assert torch.isfinite(logsnr)


def test_vae_logsnr_increases_with_signal_and_decreases_with_noise():
    mu = torch.ones(2, 3, 4)
    logvar = torch.zeros_like(mu)

    base = vae_logsnr(mu, logvar)
    higher_signal = vae_logsnr(mu * 2.0, logvar)
    higher_noise = vae_logsnr(mu, torch.full_like(logvar, math.log(4.0)))

    assert higher_signal > base
    assert higher_noise < base


def test_text_vae_encoder_returns_per_token_posterior():
    torch.manual_seed(0)
    encoder = TextVAEEncoder(
        vocab_size=23,
        latent_dim=3,
        num_layers=1,
        hidden_size=8,
        ffn_size=16,
        num_attention_heads=2,
        use_rope=False,
    )
    token_ids = torch.randint(0, 23, (2, 5))

    posterior = encoder(token_ids)

    expected_shape = (2, 5, 3)
    assert posterior.mu.shape == expected_shape
    assert posterior.logvar.shape == expected_shape


def test_text_vae_encoder_is_strictly_causal(tiny_vae_config):
    torch.manual_seed(0)
    encoder = TextVAEEncoder(config=tiny_vae_config)
    encoder.eval()
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    changed_future = token_ids.clone()
    changed_future[:, 3:] = torch.tensor([[9, 10, 11], [11, 10, 9]])

    with torch.no_grad():
        posterior = encoder(token_ids)
        changed_posterior = encoder(changed_future)

    assert torch.allclose(posterior.mu[:, :3, :], changed_posterior.mu[:, :3, :])
    assert torch.allclose(
        posterior.logvar[:, :3, :],
        changed_posterior.logvar[:, :3, :],
    )


def test_text_vae_encoder_rejects_non_causal_attention_config(tiny_vae_config):
    tiny_vae_config.attention_pattern = "full"

    with pytest.raises(ValueError, match="TextVAEEncoder requires attention_pattern"):
        TextVAEEncoder(config=tiny_vae_config)


def test_text_vae_encoder_rejects_unsupported_patch_size():
    config = VAEConfig(
        tokenizer_name="tiny",
        vocab_size=17,
        sequence_length=6,
        latent_dim=2,
        patch_size=2,
        encoder_layers=1,
        decoder_layers=1,
        hidden_size=8,
        ffn_size=16,
        num_attention_heads=2,
        attention_head_dim=4,
        use_rope=False,
    )

    with pytest.raises(NotImplementedError, match="TextVAEEncoder only supports"):
        TextVAEEncoder(config=config)


def test_text_vae_decoder_returns_token_logits(tiny_vae_config):
    torch.manual_seed(0)
    decoder = TextVAEDecoder(config=tiny_vae_config)
    token_ids = torch.randint(0, tiny_vae_config.vocab_size, (2, 5))
    latents = torch.randn(2, 5, tiny_vae_config.latent_dim)

    logits = decoder(token_ids, latents)

    assert logits.shape == (2, 5, tiny_vae_config.vocab_size)


def test_text_vae_decoder_is_strictly_causal_for_tokens_and_latents(tiny_vae_config):
    torch.manual_seed(0)
    decoder = TextVAEDecoder(config=tiny_vae_config)
    decoder.eval()
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    latents = torch.randn(2, 6, tiny_vae_config.latent_dim)
    changed_future_tokens = token_ids.clone()
    changed_future_tokens[:, 3:] = torch.tensor([[9, 10, 11], [11, 10, 9]])
    changed_future_latents = latents.clone()
    changed_future_latents[:, 3:, :] = torch.randn_like(latents[:, 3:, :]) + 10.0

    with torch.no_grad():
        logits = decoder(token_ids, latents)
        token_changed_logits = decoder(changed_future_tokens, latents)
        latent_changed_logits = decoder(token_ids, changed_future_latents)

    assert torch.allclose(logits[:, :3, :], token_changed_logits[:, :3, :])
    assert torch.allclose(logits[:, :3, :], latent_changed_logits[:, :3, :])


def test_text_vae_decoder_rejects_non_causal_attention_config(tiny_vae_config):
    tiny_vae_config.attention_pattern = "full"

    with pytest.raises(ValueError, match="TextVAEDecoder requires attention_pattern"):
        TextVAEDecoder(config=tiny_vae_config)


def test_text_vae_decoder_rejects_unsupported_patch_size():
    config = VAEConfig(
        tokenizer_name="tiny",
        vocab_size=17,
        sequence_length=6,
        latent_dim=2,
        patch_size=2,
        encoder_layers=1,
        decoder_layers=1,
        hidden_size=8,
        ffn_size=16,
        num_attention_heads=2,
        attention_head_dim=4,
        use_rope=False,
    )

    with pytest.raises(NotImplementedError, match="TextVAEDecoder only supports"):
        TextVAEDecoder(config=config)


def test_text_vae_forward_returns_posterior_latents_kl_and_logits(tiny_vae_config):
    torch.manual_seed(0)
    model = TextVAE(config=tiny_vae_config)
    generator = torch.Generator().manual_seed(1)
    token_ids = torch.randint(0, tiny_vae_config.vocab_size, (2, 6))

    output = model(token_ids, generator=generator)

    assert output.logits.shape == (2, 6, tiny_vae_config.vocab_size)
    assert output.posterior.mu.shape == (2, 6, tiny_vae_config.latent_dim)
    assert output.posterior.logvar.shape == (2, 6, tiny_vae_config.latent_dim)
    assert output.latents.shape == (2, 6, tiny_vae_config.latent_dim)
    assert output.kl.shape == (2, 6)


def test_text_vae_deterministic_mode_uses_posterior_means(tiny_vae_config):
    torch.manual_seed(0)
    model = TextVAE(config=tiny_vae_config)
    token_ids = torch.randint(0, tiny_vae_config.vocab_size, (2, 6))

    output = model(token_ids, deterministic=True)

    assert torch.allclose(output.latents, output.posterior.mu)


def test_text_vae_tiny_config_integration_with_patch_size_one():
    torch.manual_seed(0)
    config = VAEConfig(
        tokenizer_name="tiny",
        vocab_size=19,
        sequence_length=4,
        latent_dim=2,
        patch_size=1,
        encoder_layers=2,
        decoder_layers=1,
        hidden_size=8,
        ffn_size=16,
        num_attention_heads=2,
        attention_head_dim=4,
        use_rope=False,
    )
    model = TextVAE(config=config)
    model.eval()
    token_ids = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        output = model(token_ids, deterministic=True)

    assert output.logits.shape == (1, 4, config.vocab_size)
    assert output.posterior.mu.shape == (1, 4, config.latent_dim)
    assert output.posterior.logvar.shape == (1, 4, config.latent_dim)
    assert output.latents.shape == (1, 4, config.latent_dim)
    assert output.kl.shape == (1, 4)
