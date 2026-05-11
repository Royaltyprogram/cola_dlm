from dataclasses import FrozenInstanceError

import pytest
import torch
import torch.nn.functional as F

from cola_dlm.stage1 import (
    Stage1MaskingPolicy,
    Stage1VAELoss,
    apply_stage1_masking,
    compute_stage1_vae_loss,
)
from cola_dlm.vae import DiagonalGaussianPosterior, TextVAEOutput, vae_logsnr


def test_stage1_public_exports_are_loss_helpers():
    import cola_dlm.stage1 as stage1

    assert stage1.__all__ == (
        "Stage1VAELoss",
        "Stage1MaskingPolicy",
        "apply_stage1_masking",
        "compute_stage1_vae_loss",
    )


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


def test_stage1_mask_loss_matches_cross_entropy_over_mask_labels():
    logits = torch.tensor(
        [
            [[3.0, 0.0, -1.0], [0.0, 3.0, -1.0], [-1.0, 0.0, 3.0]],
            [[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        ]
    )
    token_ids = torch.tensor([[0, 1, 2], [1, 0, 2]])
    ignore_index = -1
    mask_labels = torch.tensor(
        [[ignore_index, 1, ignore_index], [1, ignore_index, 2]]
    )
    output = _make_output(logits, kl=torch.zeros_like(token_ids, dtype=logits.dtype))

    loss = compute_stage1_vae_loss(
        output,
        token_ids,
        mask_labels=mask_labels,
        lambda_kl=0.0,
        lambda_mask=0.5,
        mask_ignore_index=ignore_index,
    )

    selected = mask_labels != ignore_index
    expected_nll = F.cross_entropy(logits.reshape(-1, 3), token_ids.reshape(-1))
    expected_mask_loss = F.cross_entropy(logits[selected], mask_labels[selected])
    assert torch.allclose(loss.mask_loss, expected_mask_loss)
    assert torch.allclose(loss.loss, expected_nll + 0.5 * expected_mask_loss)


def test_stage1_mask_loss_is_zero_when_all_mask_labels_are_ignored():
    logits = torch.zeros(1, 2, 3)
    token_ids = torch.tensor([[0, 1]])
    output = _make_output(logits, kl=torch.zeros(1, 2))
    mask_labels = torch.full_like(token_ids, -100)

    loss = compute_stage1_vae_loss(
        output,
        token_ids,
        mask_labels=mask_labels,
        lambda_kl=0.0,
        lambda_mask=1.0,
    )

    assert torch.allclose(loss.mask_loss, logits.new_zeros(()))
    assert torch.isfinite(loss.loss)


def test_stage1_masking_preserves_shape_and_sets_labels_only_at_selected_positions():
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    policy = Stage1MaskingPolicy(
        mask_token_id=99,
        mask_probability=0.5,
        ignore_index=-1,
    )
    generator = torch.Generator().manual_seed(7)

    masked, labels, positions = apply_stage1_masking(
        token_ids,
        policy,
        generator=generator,
    )

    assert masked.shape == token_ids.shape
    assert labels.shape == token_ids.shape
    assert positions.shape == token_ids.shape
    assert positions.dtype == torch.bool
    assert torch.equal(masked[positions], torch.full_like(masked[positions], 99))
    assert torch.equal(masked[~positions], token_ids[~positions])
    assert torch.equal(labels[positions], token_ids[positions])
    assert torch.equal(labels[~positions], torch.full_like(labels[~positions], -1))


def test_stage1_masking_is_deterministic_with_seeded_generator():
    token_ids = torch.arange(12).reshape(2, 6)
    policy = Stage1MaskingPolicy(mask_token_id=99, mask_probability=0.4)

    first = apply_stage1_masking(
        token_ids,
        policy,
        generator=torch.Generator().manual_seed(11),
    )
    second = apply_stage1_masking(
        token_ids,
        policy,
        generator=torch.Generator().manual_seed(11),
    )

    for first_tensor, second_tensor in zip(first, second):
        assert torch.equal(first_tensor, second_tensor)


def test_stage1_masking_never_selects_attention_masked_positions():
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.tensor([[True, False, True], [False, False, True]])
    policy = Stage1MaskingPolicy(
        mask_token_id=99,
        mask_probability=1.0,
        ignore_index=-1,
    )

    masked, labels, positions = apply_stage1_masking(
        token_ids,
        policy,
        attention_mask=attention_mask,
    )

    assert torch.equal(positions, attention_mask)
    assert torch.equal(masked[~attention_mask], token_ids[~attention_mask])
    assert torch.equal(
        labels[~attention_mask],
        torch.full_like(labels[~attention_mask], -1),
    )


def test_stage1_masking_validates_policy_and_token_ids():
    with pytest.raises(ValueError, match="mask_token_id"):
        Stage1MaskingPolicy(mask_token_id=-1, mask_probability=0.5)
    with pytest.raises(ValueError, match="mask_probability"):
        Stage1MaskingPolicy(mask_token_id=1, mask_probability=1.1)
    with pytest.raises(ValueError, match="non-negative"):
        apply_stage1_masking(
            torch.tensor([[1, -2]]),
            Stage1MaskingPolicy(mask_token_id=99, mask_probability=0.5),
        )


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
