from dataclasses import FrozenInstanceError

import pytest
import torch
import torch.nn.functional as F

from cola_dlm.stage1 import Stage1VAELoss, compute_stage1_vae_loss
from cola_dlm.vae import DiagonalGaussianPosterior, TextVAEOutput, vae_logsnr


def test_stage1_public_exports_are_loss_helpers():
    import cola_dlm.stage1 as stage1

    assert stage1.__all__ == ("Stage1VAELoss", "compute_stage1_vae_loss")


def test_stage1_reconstruction_nll_matches_cross_entropy():
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]]])
    token_ids = torch.tensor([[0, 2]])
    output = _make_output(logits, kl=torch.zeros(1, 2))

    loss = compute_stage1_vae_loss(output, token_ids, lambda_kl=0.0)

    expected = F.cross_entropy(logits.reshape(-1, 3), token_ids.reshape(-1))
    assert torch.allclose(loss.reconstruction_nll, expected)
    assert torch.allclose(loss.loss, expected)


def test_stage1_kl_is_valid_token_average_and_weighted_in_loss():
    logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 3.0], [10.0, -10.0]],
            [[1.0, 1.0], [-10.0, 10.0], [10.0, -10.0]],
        ]
    )
    token_ids = torch.tensor([[0, 1, 1], [0, 0, 1]])
    attention_mask = torch.tensor(
        [[True, True, False], [True, False, False]]
    )
    kl = torch.tensor([[1.0, 2.0, 100.0], [4.0, 100.0, 100.0]])
    output = _make_output(logits, kl=kl)

    loss = compute_stage1_vae_loss(
        output,
        token_ids,
        attention_mask=attention_mask,
        lambda_kl=0.25,
    )

    per_token_nll = F.cross_entropy(
        logits.reshape(-1, 2),
        token_ids.reshape(-1),
        reduction="none",
    ).reshape(token_ids.shape)
    expected_nll = per_token_nll[attention_mask].mean()
    expected_kl = torch.tensor((1.0 + 2.0 + 4.0) / 3.0)
    assert torch.allclose(loss.reconstruction_nll, expected_nll)
    assert torch.allclose(loss.kl, expected_kl)
    assert torch.allclose(loss.loss, expected_nll + 0.25 * expected_kl)


def test_stage1_loss_returns_scalar_frozen_output_fields():
    logits = torch.zeros(1, 2, 3)
    token_ids = torch.tensor([[0, 1]])
    mu = torch.ones(1, 2, 4)
    logvar = torch.zeros_like(mu)
    output = _make_output(
        logits,
        kl=torch.tensor([[0.5, 1.5]]),
        mu=mu,
        logvar=logvar,
    )

    loss = compute_stage1_vae_loss(output, token_ids)

    assert isinstance(loss, Stage1VAELoss)
    for value in (
        loss.loss,
        loss.reconstruction_nll,
        loss.kl,
        loss.mask_loss,
        loss.logsnr,
    ):
        assert value.shape == ()
    assert torch.allclose(loss.logsnr, vae_logsnr(mu, logvar))
    with pytest.raises(FrozenInstanceError):
        loss.loss = torch.tensor(0.0)


def test_stage1_mask_loss_is_zero_when_lambda_mask_is_zero():
    logits = torch.zeros(1, 2, 3)
    token_ids = torch.tensor([[0, 1]])
    output = _make_output(logits, kl=torch.zeros(1, 2))
    mask_labels = torch.tensor([[0, 1]])

    loss = compute_stage1_vae_loss(
        output,
        token_ids,
        mask_labels=mask_labels,
        lambda_kl=0.0,
        lambda_mask=0.0,
    )

    expected_nll = F.cross_entropy(logits.reshape(-1, 3), token_ids.reshape(-1))
    assert torch.allclose(loss.mask_loss, logits.new_zeros(()))
    assert torch.allclose(loss.loss, expected_nll)


def _make_output(
    logits: torch.Tensor,
    *,
    kl: torch.Tensor,
    mu: torch.Tensor | None = None,
    logvar: torch.Tensor | None = None,
) -> TextVAEOutput:
    if mu is None:
        mu = torch.zeros(*logits.shape[:2], 2, dtype=logits.dtype)
    if logvar is None:
        logvar = torch.zeros_like(mu)

    posterior = DiagonalGaussianPosterior(mu=mu, logvar=logvar)
    return TextVAEOutput(
        logits=logits,
        posterior=posterior,
        latents=posterior.mode(),
        kl=kl,
    )
