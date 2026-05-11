from dataclasses import FrozenInstanceError

import pytest
import torch
import torch.nn.functional as F

from cola_dlm.stage1 import (
    Stage1MaskingPolicy,
    Stage1VAELoss,
    apply_stage1_masking,
    compute_stage1_vae_loss,
    stage1_pretraining_step,
)
from cola_dlm.vae import (
    DiagonalGaussianPosterior,
    TextVAE,
    TextVAEOutput,
    vae_logsnr,
)


def test_stage1_public_exports_are_loss_helpers():
    import cola_dlm.stage1 as stage1

    assert stage1.__all__ == (
        "Stage1VAELoss",
        "Stage1MaskingPolicy",
        "apply_stage1_masking",
        "compute_stage1_vae_loss",
        "stage1_pretraining_step",
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


def test_stage1_config_weights_map_to_lambda_names(tiny_stage1_config):
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    token_ids = torch.tensor([[0, 1]])
    kl = torch.tensor([[2.0, 4.0]])
    mask_labels = torch.tensor([[-100, 1]])
    output = _make_output(logits, kl=kl)

    loss = compute_stage1_vae_loss(
        output,
        token_ids,
        mask_labels=mask_labels,
        stage1_config=tiny_stage1_config,
    )

    selected = mask_labels != -100
    expected_nll = F.cross_entropy(logits.reshape(-1, 2), token_ids.reshape(-1))
    expected_mask_loss = F.cross_entropy(logits[selected], mask_labels[selected])
    expected_loss = (
        expected_nll
        + tiny_stage1_config.kl_weight * kl.mean()
        + tiny_stage1_config.mask_loss_weight * expected_mask_loss
    )
    assert torch.allclose(loss.loss, expected_loss)

    overridden = compute_stage1_vae_loss(
        output,
        token_ids,
        mask_labels=mask_labels,
        stage1_config=tiny_stage1_config,
        lambda_kl=0.0,
        lambda_mask=0.0,
    )
    assert torch.allclose(overridden.loss, expected_nll)


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
    for value in loss.as_dict().values():
        assert value.shape == ()
    assert torch.allclose(loss.logsnr, vae_logsnr(mu, logvar))
    with pytest.raises(FrozenInstanceError):
        loss.loss = torch.tensor(0.0)


def test_stage1_loss_as_dict_uses_stable_diagnostic_names():
    logits = torch.zeros(1, 2, 3)
    token_ids = torch.tensor([[0, 1]])
    output = _make_output(logits, kl=torch.zeros(1, 2))

    loss = compute_stage1_vae_loss(output, token_ids)
    diagnostics = loss.as_dict()

    assert tuple(diagnostics) == (
        "loss",
        "reconstruction_nll",
        "kl",
        "mask_loss",
        "logsnr",
    )
    assert diagnostics["loss"] is loss.loss
    assert diagnostics["reconstruction_nll"] is loss.reconstruction_nll
    assert diagnostics["kl"] is loss.kl
    assert diagnostics["mask_loss"] is loss.mask_loss
    assert diagnostics["logsnr"] is loss.logsnr


def test_stage1_loss_components_are_scalar_on_output_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.zeros(1, 2, 3, device=device)
    token_ids = torch.tensor([[0, 1]], device=device)
    output = _make_output(logits, kl=torch.zeros(1, 2, device=device))

    loss = compute_stage1_vae_loss(output, token_ids)

    for name, value in loss.as_dict().items():
        assert value.shape == (), name
        assert value.device == logits.device, name


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


def test_stage1_pretraining_step_updates_parameter_and_returns_finite_loss(
    tiny_vae_config,
):
    torch.manual_seed(0)
    model = TextVAE(config=tiny_vae_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    token_ids = torch.arange(12, dtype=torch.long).reshape(2, 6)
    policy = Stage1MaskingPolicy(
        mask_token_id=tiny_vae_config.vocab_size - 1,
        mask_probability=1.0,
    )
    parameters = [param for param in model.parameters() if param.requires_grad]
    before_step = [param.detach().clone() for param in parameters]

    loss = stage1_pretraining_step(
        model,
        optimizer,
        token_ids,
        masking_policy=policy,
        lambda_kl=0.1,
        lambda_mask=0.2,
        generator=torch.Generator().manual_seed(1),
        max_grad_norm=1.0,
    )

    for value in (
        loss.loss,
        loss.reconstruction_nll,
        loss.kl,
        loss.mask_loss,
        loss.logsnr,
    ):
        assert value.shape == ()
        assert torch.isfinite(value)
    assert any(
        not torch.allclose(before, after.detach())
        for before, after in zip(before_step, parameters)
    )


def test_stage1_pretraining_step_skips_masking_when_lambda_mask_is_zero(
    tiny_vae_config,
    tiny_stage1_config,
):
    torch.manual_seed(0)
    model = TextVAE(config=tiny_vae_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    token_ids = torch.arange(12, dtype=torch.long).reshape(2, 6)

    loss = stage1_pretraining_step(
        model,
        optimizer,
        token_ids,
        stage1_config=tiny_stage1_config,
        lambda_kl=0.0,
        lambda_mask=0.0,
        generator=torch.Generator().manual_seed(1),
    )

    assert torch.allclose(loss.mask_loss, loss.mask_loss.new_zeros(()))


def _make_output(
    logits: torch.Tensor,
    *,
    kl: torch.Tensor,
    mu: torch.Tensor | None = None,
    logvar: torch.Tensor | None = None,
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

    posterior = DiagonalGaussianPosterior(mu=mu, logvar=logvar)
    return TextVAEOutput(
        logits=logits,
        posterior=posterior,
        latents=posterior.mode(),
        kl=kl,
    )
