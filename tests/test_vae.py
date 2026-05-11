import math

import pytest
import torch

from cola_dlm.vae import DiagonalGaussianPosterior, vae_logsnr


def test_vae_public_exports_are_core_utilities():
    import cola_dlm.vae as vae

    assert vae.__all__ == (
        "DiagonalGaussianPosterior",
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
