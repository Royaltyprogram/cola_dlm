from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from cola_dlm.diagnostics import VAEDiagnostics, compute_vae_diagnostics
from cola_dlm.vae import DiagonalGaussianPosterior, TextVAEOutput, vae_logsnr


def test_diagnostics_public_exports_are_small():
    import cola_dlm.diagnostics as diagnostics

    assert diagnostics.__all__ == ("VAEDiagnostics", "compute_vae_diagnostics")


def test_reconstruction_accuracy_counts_top1_predictions():
    logits = torch.tensor(
        [[[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]]
    )
    token_ids = torch.tensor([[0, 2, 2]])
    output = _make_output(logits)

    diagnostics = compute_vae_diagnostics(output, token_ids)

    assert torch.allclose(
        diagnostics.reconstruction_accuracy,
        torch.tensor(2.0 / 3.0),
    )


def test_reconstruction_accuracy_respects_attention_mask():
    logits = torch.tensor(
        [[[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]]
    )
    token_ids = torch.tensor([[0, 2, 2]])
    attention_mask = torch.tensor([[True, False, True]])
    output = _make_output(logits)

    diagnostics = compute_vae_diagnostics(
        output,
        token_ids,
        attention_mask=attention_mask,
    )

    assert torch.allclose(diagnostics.reconstruction_accuracy, torch.tensor(1.0))


def test_logsnr_matches_vae_helper():
    logits = torch.zeros(1, 2, 3)
    token_ids = torch.tensor([[0, 1]])
    mu = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    logvar = torch.log(torch.tensor([[[0.5, 1.0], [2.0, 4.0]]]))
    output = _make_output(logits, mu=mu, logvar=logvar)

    diagnostics = compute_vae_diagnostics(output, token_ids)

    assert torch.allclose(diagnostics.logsnr, vae_logsnr(mu, logvar))


def test_latent_norm_and_posterior_variance_values_are_known():
    logits = torch.zeros(2, 2, 3)
    token_ids = torch.tensor([[0, 1], [2, 0]])
    latents = torch.tensor(
        [
            [[3.0, 4.0], [0.0, 0.0]],
            [[5.0, 12.0], [8.0, 15.0]],
        ]
    )
    variances = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    output = _make_output(
        logits,
        mu=torch.ones_like(latents),
        logvar=torch.log(variances),
        latents=latents,
    )

    diagnostics = compute_vae_diagnostics(output, token_ids)

    expected_norms = torch.tensor([5.0, 0.0, 13.0, 17.0])
    assert torch.allclose(diagnostics.latent_norm_mean, expected_norms.mean())
    assert torch.allclose(
        diagnostics.latent_norm_std,
        expected_norms.std(unbiased=False),
    )
    assert torch.allclose(diagnostics.posterior_variance_mean, variances.mean())
    assert torch.allclose(
        diagnostics.posterior_variance_std,
        variances.std(unbiased=False),
    )


def test_diagnostics_as_dict_uses_stable_flat_names():
    logits = torch.zeros(1, 2, 3)
    token_ids = torch.tensor([[0, 1]])
    output = _make_output(logits)

    diagnostics = compute_vae_diagnostics(output, token_ids)
    values = diagnostics.as_dict()

    assert tuple(values) == (
        "reconstruction_accuracy",
        "logsnr",
        "latent_norm_mean",
        "latent_norm_std",
        "posterior_variance_mean",
        "posterior_variance_std",
    )
    assert values["reconstruction_accuracy"] is diagnostics.reconstruction_accuracy
    assert values["logsnr"] is diagnostics.logsnr
    assert values["latent_norm_mean"] is diagnostics.latent_norm_mean
    assert values["latent_norm_std"] is diagnostics.latent_norm_std
    assert values["posterior_variance_mean"] is diagnostics.posterior_variance_mean
    assert values["posterior_variance_std"] is diagnostics.posterior_variance_std


def test_returned_diagnostics_are_scalar_detached_tensors():
    logits = torch.tensor(
        [[[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]]],
        requires_grad=True,
    )
    token_ids = torch.tensor([[0, 1]])
    mu = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    logvar = torch.full((1, 2, 2), math.log(2.0), requires_grad=True)
    latents = torch.tensor([[[3.0, 4.0], [5.0, 12.0]]], requires_grad=True)
    output = _make_output(logits, mu=mu, logvar=logvar, latents=latents)

    diagnostics = compute_vae_diagnostics(output, token_ids)

    assert isinstance(diagnostics, VAEDiagnostics)
    for name, value in diagnostics.as_dict().items():
        assert value.shape == (), name
        assert not value.requires_grad, name
        assert value.grad_fn is None, name
    with pytest.raises(FrozenInstanceError):
        diagnostics.logsnr = torch.tensor(0.0)


def _make_output(
    logits: torch.Tensor,
    *,
    mu: torch.Tensor | None = None,
    logvar: torch.Tensor | None = None,
    latents: torch.Tensor | None = None,
) -> TextVAEOutput:
    if mu is None:
        mu = torch.zeros(
            *logits.shape[:2],
            2,
            dtype=logits.dtype,
            device=logits.device,
        )
    if logvar is None:
        logvar = torch.zeros_like(mu)
    if latents is None:
        latents = mu

    posterior = DiagonalGaussianPosterior(mu=mu, logvar=logvar)
    return TextVAEOutput(
        logits=logits,
        posterior=posterior,
        latents=latents,
        kl=torch.zeros(logits.shape[:2], dtype=logits.dtype, device=logits.device),
    )
