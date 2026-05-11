from dataclasses import FrozenInstanceError, replace

import pytest
import torch
import torch.nn.functional as F

from cola_dlm.block_causal_mask import build_packed_dit_inputs
from cola_dlm.config import Stage2Config
from cola_dlm.dit import BlockCausalTextDiT
from cola_dlm.stage2 import (
    Stage2Loss,
    compute_stage2_loss,
    compute_stage2_vae_loss,
    create_frozen_reference_encoder,
    reference_kl,
    stage2_joint_training_step,
)
from cola_dlm.vae import (
    DiagonalGaussianPosterior,
    TextVAE,
    TextVAEEncoder,
    TextVAEOutput,
)


def test_stage2_public_surface_starts_with_reference_encoder_helper():
    import cola_dlm.stage2 as stage2

    assert stage2.__all__ == (
        "Stage2Loss",
        "create_frozen_reference_encoder",
        "reference_kl",
        "compute_stage2_vae_loss",
        "compute_stage2_loss",
        "stage2_joint_training_step",
    )
    assert stage2.Stage2Loss is Stage2Loss
    assert stage2.create_frozen_reference_encoder is create_frozen_reference_encoder
    assert stage2.reference_kl is reference_kl
    assert stage2.compute_stage2_vae_loss is compute_stage2_vae_loss
    assert stage2.compute_stage2_loss is compute_stage2_loss
    assert stage2.stage2_joint_training_step is stage2_joint_training_step


def test_create_frozen_reference_encoder_copies_values_without_shared_storage(
    tiny_vae_config,
):
    torch.manual_seed(0)
    vae = TextVAE(tiny_vae_config)

    reference_encoder = create_frozen_reference_encoder(vae)

    assert isinstance(reference_encoder, TextVAEEncoder)
    reference_parameters = dict(reference_encoder.named_parameters())
    for name, parameter in vae.encoder.named_parameters():
        reference_parameter = reference_parameters[name]
        torch.testing.assert_close(reference_parameter, parameter)
        assert reference_parameter is not parameter
        assert (
            reference_parameter.untyped_storage().data_ptr()
            != parameter.untyped_storage().data_ptr()
        )


def test_create_frozen_reference_encoder_disables_training_and_gradients(
    tiny_vae_config,
):
    vae = TextVAE(tiny_vae_config)
    vae.train()

    reference_encoder = create_frozen_reference_encoder(vae)

    assert reference_encoder.training is False
    assert all(
        not parameter.requires_grad for parameter in reference_encoder.parameters()
    )


def test_create_frozen_reference_encoder_is_unchanged_by_trainable_encoder_updates(
    tiny_vae_config,
):
    torch.manual_seed(0)
    vae = TextVAE(tiny_vae_config)
    reference_encoder = create_frozen_reference_encoder(vae)
    reference_parameters = {
        name: parameter.detach().clone()
        for name, parameter in reference_encoder.named_parameters()
    }

    with torch.no_grad():
        for parameter in vae.encoder.parameters():
            parameter.add_(1.0)

    trainable_parameters = dict(vae.encoder.named_parameters())
    for name, reference_parameter in reference_encoder.named_parameters():
        torch.testing.assert_close(reference_parameter, reference_parameters[name])
        trainable_parameter = trainable_parameters[name]
        assert not torch.allclose(reference_parameter, trainable_parameter)


def test_private_stage2_shape_validation_accepts_matching_components(
    tiny_stage2_config: Stage2Config,
):
    import cola_dlm.stage2 as stage2

    vae = TextVAE(tiny_stage2_config.vae)
    reference_encoder = create_frozen_reference_encoder(vae)
    dit = BlockCausalTextDiT(tiny_stage2_config.dit)

    stage2._validate_stage2_component_shapes(
        vae=vae,
        reference_encoder=reference_encoder,
        dit=dit,
        stage2_config=tiny_stage2_config,
    )


def test_private_stage2_shape_validation_rejects_mismatched_dit_latent_dim(
    tiny_stage2_config: Stage2Config,
):
    import cola_dlm.stage2 as stage2

    vae = TextVAE(tiny_stage2_config.vae)
    reference_encoder = create_frozen_reference_encoder(vae)
    dit_config = replace(
        tiny_stage2_config.dit,
        latent_dim=tiny_stage2_config.dit.latent_dim + 1,
    )
    dit = BlockCausalTextDiT(dit_config)

    with pytest.raises(ValueError, match="dit.config.latent_dim"):
        stage2._validate_stage2_component_shapes(
            vae=vae,
            reference_encoder=reference_encoder,
            dit=dit,
            stage2_config=tiny_stage2_config,
        )


def test_stage2_loss_returns_scalar_frozen_output_fields():
    logits = torch.zeros(1, 2, 3)
    token_ids = torch.tensor([[0, 1]])
    output = _make_output(logits)
    reference_posterior = _make_posterior(torch.zeros(1, 2, 2))

    loss = compute_stage2_vae_loss(output, token_ids, reference_posterior)

    assert isinstance(loss, Stage2Loss)
    for name, value in loss.as_dict().items():
        assert value.shape == (), name
        assert torch.isfinite(value), name
    with pytest.raises(FrozenInstanceError):
        loss.loss = torch.tensor(0.0)


def test_stage2_loss_as_dict_uses_stable_diagnostic_names():
    logits = torch.zeros(1, 2, 3)
    token_ids = torch.tensor([[0, 1]])
    output = _make_output(logits)
    reference_posterior = _make_posterior(torch.zeros(1, 2, 2))

    loss = compute_stage2_vae_loss(output, token_ids, reference_posterior)
    diagnostics = loss.as_dict()

    assert tuple(diagnostics) == (
        "loss",
        "vae_loss",
        "flow_matching_loss",
        "reference_kl",
        "reconstruction_nll",
        "posterior_regularizer",
        "mask_loss",
        "logsnr",
        "reconstruction_accuracy",
        "latent_norm_mean",
        "latent_norm_std",
        "posterior_variance_mean",
        "posterior_variance_std",
    )
    assert diagnostics["loss"] is loss.loss
    assert diagnostics["vae_loss"] is loss.vae_loss
    assert diagnostics["flow_matching_loss"] is loss.flow_matching_loss
    assert diagnostics["reference_kl"] is loss.reference_kl
    assert diagnostics["reconstruction_nll"] is loss.reconstruction_nll
    assert diagnostics["posterior_regularizer"] is loss.posterior_regularizer
    assert diagnostics["mask_loss"] is loss.mask_loss
    assert diagnostics["logsnr"] is loss.logsnr
    assert (
        diagnostics["reconstruction_accuracy"]
        is loss.diagnostics.reconstruction_accuracy
    )
    assert diagnostics["latent_norm_mean"] is loss.diagnostics.latent_norm_mean
    assert diagnostics["latent_norm_std"] is loss.diagnostics.latent_norm_std
    assert (
        diagnostics["posterior_variance_mean"]
        is loss.diagnostics.posterior_variance_mean
    )
    assert (
        diagnostics["posterior_variance_std"]
        is loss.diagnostics.posterior_variance_std
    )


def test_stage2_reconstruction_nll_matches_cross_entropy():
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]]])
    token_ids = torch.tensor([[0, 2]])
    output = _make_output(logits)
    reference_posterior = _make_posterior(torch.zeros(1, 2, 2))

    loss = compute_stage2_vae_loss(
        output,
        token_ids,
        reference_posterior,
        lambda_posterior_regularizer=0.0,
    )

    expected = F.cross_entropy(logits.reshape(-1, 3), token_ids.reshape(-1))
    assert torch.allclose(loss.reconstruction_nll, expected)


def test_reference_kl_is_zero_for_identical_posteriors():
    posterior = _make_posterior(
        torch.tensor([[[0.0, 1.0], [2.0, -1.0]]]),
        logvar=torch.tensor([[[0.5, -0.5], [0.25, -0.25]]]),
    )

    loss = reference_kl(posterior, posterior)

    assert torch.allclose(loss, loss.new_zeros(()))


def test_reference_kl_is_positive_for_shifted_trainable_posterior():
    trainable_posterior = _make_posterior(torch.ones(1, 2, 2))
    reference_posterior = _make_posterior(torch.zeros(1, 2, 2))

    loss = reference_kl(trainable_posterior, reference_posterior)

    assert loss.item() > 0.0
    assert torch.allclose(loss, torch.tensor(1.0))


def test_attention_mask_restricts_reconstruction_and_reference_kl():
    logits = torch.tensor(
        [[[3.0, 0.0], [-10.0, 10.0], [0.0, 3.0]]],
    )
    token_ids = torch.tensor([[0, 0, 1]])
    attention_mask = torch.tensor([[True, False, True]])
    trainable_posterior = _make_posterior(torch.tensor([[[1.0], [100.0], [3.0]]]))
    reference_posterior = _make_posterior(torch.zeros(1, 3, 1))
    output = _make_output(logits, posterior=trainable_posterior)

    loss = compute_stage2_vae_loss(
        output,
        token_ids,
        reference_posterior,
        attention_mask=attention_mask,
        lambda_posterior_regularizer=0.0,
    )

    per_token_nll = F.cross_entropy(
        logits.reshape(-1, 2),
        token_ids.reshape(-1),
        reduction="none",
    ).reshape(token_ids.shape)
    expected_nll = per_token_nll[attention_mask].mean()
    expected_kl = torch.tensor((0.5 * 1.0**2 + 0.5 * 3.0**2) / 2.0)
    assert torch.allclose(loss.reconstruction_nll, expected_nll)
    assert torch.allclose(loss.reference_kl, expected_kl)
    assert torch.allclose(loss.diagnostics.reconstruction_accuracy, torch.tensor(1.0))


def test_stage2_mask_loss_handles_ignored_and_selected_labels():
    logits = torch.tensor(
        [
            [[3.0, 0.0, -1.0], [0.0, 3.0, -1.0], [-1.0, 0.0, 3.0]],
            [[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        ]
    )
    token_ids = torch.tensor([[0, 1, 2], [1, 0, 2]])
    output = _make_output(logits, latent_dim=2)
    reference_posterior = _make_posterior(torch.zeros(2, 3, 2))
    ignored_labels = torch.full_like(token_ids, -100)

    ignored_loss = compute_stage2_vae_loss(
        output,
        token_ids,
        reference_posterior,
        mask_labels=ignored_labels,
        lambda_mask=1.0,
    )

    mask_labels = torch.tensor([[-100, 1, -100], [1, -100, 2]])
    selected_loss = compute_stage2_vae_loss(
        output,
        token_ids,
        reference_posterior,
        mask_labels=mask_labels,
        lambda_mask=1.0,
    )

    selected = mask_labels != -100
    expected_mask_loss = F.cross_entropy(logits[selected], mask_labels[selected])
    assert torch.allclose(ignored_loss.mask_loss, logits.new_zeros(()))
    assert torch.isfinite(selected_loss.mask_loss)
    assert torch.allclose(selected_loss.mask_loss, expected_mask_loss)


def test_stage2_config_weights_and_explicit_overrides_control_total_loss(
    tiny_stage2_config: Stage2Config,
):
    logits = torch.zeros(1, 2, 3)
    token_ids = torch.tensor([[0, 1]])
    trainable_posterior = _make_posterior(torch.ones(1, 2, 2))
    reference_posterior = _make_posterior(torch.zeros(1, 2, 2))
    output = _make_output(logits, posterior=trainable_posterior)

    configured = compute_stage2_vae_loss(
        output,
        token_ids,
        reference_posterior,
        stage2_config=tiny_stage2_config,
        lambda_posterior_regularizer=0.0,
    )
    expected = (
        tiny_stage2_config.vae_loss_weight * configured.vae_loss
        + tiny_stage2_config.reference_kl_weight * configured.reference_kl
    )
    assert torch.allclose(configured.loss, expected)

    overridden = compute_stage2_vae_loss(
        output,
        token_ids,
        reference_posterior,
        stage2_config=tiny_stage2_config,
        lambda_vae=0.0,
        lambda_reference_kl=0.0,
        lambda_posterior_regularizer=0.0,
    )
    assert torch.allclose(overridden.loss, logits.new_zeros(()))

    with pytest.raises(ValueError, match="lambda_reference_kl"):
        compute_stage2_vae_loss(
            output,
            token_ids,
            reference_posterior,
            lambda_reference_kl=-1.0,
        )


def test_compute_stage2_loss_returns_finite_scalar_components(
    tiny_stage2_config: Stage2Config,
):
    vae, reference_encoder, dit = _make_stage2_modules(tiny_stage2_config)
    token_ids = _make_stage2_token_ids(tiny_stage2_config)

    loss = compute_stage2_loss(
        vae,
        reference_encoder,
        dit,
        token_ids,
        stage2_config=tiny_stage2_config,
        generator=torch.Generator().manual_seed(101),
    )

    for name, value in loss.as_dict().items():
        assert value.shape == (), name
        assert torch.isfinite(value), name
    assert loss.flow_matching_loss.item() >= 0.0
    expected_block_ids = tuple(
        range(
            tiny_stage2_config.dit.sequence_length
            // tiny_stage2_config.dit.block_size
        )
    )
    assert tuple(loss.flow_matching_loss_by_block) == expected_block_ids
    metrics = loss.as_dict()
    for block_id in expected_block_ids:
        key = f"flow_matching_loss_block_{block_id}"
        assert key in metrics
        assert torch.isfinite(metrics[key])


def test_stage2_packed_clean_history_is_detached_but_noisy_targets_backpropagate(
    tiny_stage2_config: Stage2Config,
):
    z0 = torch.randn(
        2,
        tiny_stage2_config.dit.sequence_length,
        tiny_stage2_config.dit.latent_dim,
        requires_grad=True,
    )
    zt = z0 * 2.0

    packed = build_packed_dit_inputs(
        z0,
        zt,
        block_size=tiny_stage2_config.dit.block_size,
    )

    clean_length = tiny_stage2_config.dit.sequence_length
    clean_length -= tiny_stage2_config.dit.block_size
    clean_probe = packed.latents[:, :clean_length].square().sum()
    noisy_probe = packed.latents[:, clean_length:].square().sum()
    clean_grad = None
    if clean_probe.requires_grad:
        clean_grad = torch.autograd.grad(
            clean_probe,
            z0,
            allow_unused=True,
            retain_graph=True,
        )[0]
    noisy_grad = torch.autograd.grad(noisy_probe, z0, allow_unused=True)[0]

    if clean_grad is not None:
        torch.testing.assert_close(clean_grad, torch.zeros_like(z0))
    assert noisy_grad is not None
    assert torch.count_nonzero(noisy_grad).item() > 0


def test_compute_stage2_loss_backward_reaches_trainable_vae_and_dit(
    tiny_stage2_config: Stage2Config,
):
    vae, reference_encoder, dit = _make_stage2_modules(tiny_stage2_config)
    token_ids = _make_stage2_token_ids(tiny_stage2_config)

    loss = compute_stage2_loss(
        vae,
        reference_encoder,
        dit,
        token_ids,
        stage2_config=tiny_stage2_config,
        generator=torch.Generator().manual_seed(103),
    )
    loss.loss.backward()

    assert _has_nonzero_grad(vae.parameters())
    assert _has_nonzero_grad(dit.parameters())


def test_compute_stage2_loss_backward_leaves_reference_encoder_without_grad(
    tiny_stage2_config: Stage2Config,
):
    vae, reference_encoder, dit = _make_stage2_modules(tiny_stage2_config)
    token_ids = _make_stage2_token_ids(tiny_stage2_config)

    loss = compute_stage2_loss(
        vae,
        reference_encoder,
        dit,
        token_ids,
        stage2_config=tiny_stage2_config,
        generator=torch.Generator().manual_seed(107),
    )
    loss.loss.backward()

    assert all(parameter.grad is None for parameter in reference_encoder.parameters())


@pytest.mark.parametrize("prediction_type", ["velocity", "x0"])
def test_compute_stage2_loss_prediction_type_paths_are_finite(
    tiny_stage2_config: Stage2Config,
    prediction_type: str,
):
    config = replace(
        tiny_stage2_config,
        diffusion=replace(
            tiny_stage2_config.diffusion,
            prediction_type=prediction_type,
        ),
    )
    vae, reference_encoder, dit = _make_stage2_modules(config)
    token_ids = _make_stage2_token_ids(config)

    loss = compute_stage2_loss(
        vae,
        reference_encoder,
        dit,
        token_ids,
        stage2_config=config,
        generator=torch.Generator().manual_seed(109),
    )

    assert torch.isfinite(loss.loss)
    assert torch.isfinite(loss.flow_matching_loss)


def test_stage2_joint_training_step_updates_trainable_modules_only(
    tiny_stage2_config: Stage2Config,
):
    vae, reference_encoder, dit = _make_stage2_modules(tiny_stage2_config)
    reference_encoder.train()
    optimizer = torch.optim.AdamW(
        list(vae.parameters()) + list(dit.parameters()),
        lr=1.0e-3,
        weight_decay=0.0,
    )
    token_ids = _make_stage2_token_ids(tiny_stage2_config)
    vae_parameters = [parameter for parameter in vae.parameters()]
    dit_parameters = [parameter for parameter in dit.parameters()]
    reference_parameters = [parameter for parameter in reference_encoder.parameters()]
    vae_before = _clone_parameters(vae_parameters)
    dit_before = _clone_parameters(dit_parameters)
    reference_before = _clone_parameters(reference_parameters)
    for parameter in reference_parameters:
        parameter.grad = torch.ones_like(parameter)

    loss = stage2_joint_training_step(
        vae,
        reference_encoder,
        dit,
        optimizer,
        token_ids,
        stage2_config=tiny_stage2_config,
        generator=torch.Generator().manual_seed(113),
        max_grad_norm=1.0,
    )

    assert torch.isfinite(loss.loss)
    assert _has_changed_parameter(vae_before, vae_parameters)
    assert _has_changed_parameter(dit_before, dit_parameters)
    for before, after in zip(reference_before, reference_parameters):
        torch.testing.assert_close(before, after)
        assert after.grad is None
    assert reference_encoder.training is False


def test_stage2_joint_training_step_validates_masks_and_weights(
    tiny_stage2_config: Stage2Config,
):
    vae, reference_encoder, dit = _make_stage2_modules(tiny_stage2_config)
    optimizer = torch.optim.AdamW(
        list(vae.parameters()) + list(dit.parameters()),
        lr=1.0e-3,
    )
    token_ids = _make_stage2_token_ids(tiny_stage2_config)
    bad_attention_mask = torch.ones(
        token_ids.shape[0],
        token_ids.shape[1] + 1,
        dtype=torch.bool,
    )

    with pytest.raises(ValueError, match="attention_mask must match token batch shape"):
        stage2_joint_training_step(
            vae,
            reference_encoder,
            dit,
            optimizer,
            token_ids,
            attention_mask=bad_attention_mask,
            stage2_config=tiny_stage2_config,
        )

    with pytest.raises(ValueError, match="lambda_flow_matching must be non-negative"):
        stage2_joint_training_step(
            vae,
            reference_encoder,
            dit,
            optimizer,
            token_ids,
            stage2_config=tiny_stage2_config,
            lambda_flow_matching=-0.1,
        )


def test_compute_stage2_loss_validates_token_ids(
    tiny_stage2_config: Stage2Config,
):
    vae, reference_encoder, dit = _make_stage2_modules(tiny_stage2_config)
    token_ids = _make_stage2_token_ids(tiny_stage2_config)

    with pytest.raises(ValueError, match="token_ids must be a torch.long tensor"):
        compute_stage2_loss(
            vae,
            reference_encoder,
            dit,
            token_ids.float(),
            stage2_config=tiny_stage2_config,
        )

    negative_token_ids = token_ids.clone()
    negative_token_ids[0, 0] = -1
    with pytest.raises(ValueError, match="token_ids must be non-negative"):
        compute_stage2_loss(
            vae,
            reference_encoder,
            dit,
            negative_token_ids,
            stage2_config=tiny_stage2_config,
        )

    too_large_token_ids = token_ids.clone()
    too_large_token_ids[0, 0] = tiny_stage2_config.vae.vocab_size
    with pytest.raises(ValueError, match="token_ids must be less than vocab_size"):
        compute_stage2_loss(
            vae,
            reference_encoder,
            dit,
            too_large_token_ids,
            stage2_config=tiny_stage2_config,
        )


def _make_output(
    logits: torch.Tensor,
    *,
    posterior: DiagonalGaussianPosterior | None = None,
    latent_dim: int = 2,
) -> TextVAEOutput:
    if posterior is None:
        posterior = _make_posterior(
            torch.zeros(
                *logits.shape[:2],
                latent_dim,
                dtype=logits.dtype,
                device=logits.device,
            )
        )
    return TextVAEOutput(
        logits=logits,
        posterior=posterior,
        latents=posterior.mode(),
        kl=posterior.kl(),
    )


def _make_posterior(
    mu: torch.Tensor,
    *,
    logvar: torch.Tensor | None = None,
) -> DiagonalGaussianPosterior:
    if logvar is None:
        logvar = torch.zeros_like(mu)
    return DiagonalGaussianPosterior(mu=mu, logvar=logvar)


def _make_stage2_modules(
    config: Stage2Config,
) -> tuple[TextVAE, TextVAEEncoder, BlockCausalTextDiT]:
    torch.manual_seed(0)
    vae = TextVAE(config.vae)
    reference_encoder = create_frozen_reference_encoder(vae)
    dit = BlockCausalTextDiT(config.dit)
    return vae, reference_encoder, dit


def _make_stage2_token_ids(config: Stage2Config) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    return torch.randint(
        0,
        config.vae.vocab_size,
        (2, config.vae.sequence_length),
        generator=generator,
    )


def _has_nonzero_grad(parameters) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in parameters
    )


def _clone_parameters(parameters) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _has_changed_parameter(
    before_parameters: list[torch.Tensor],
    after_parameters,
) -> bool:
    return any(
        not torch.allclose(before, after.detach())
        for before, after in zip(before_parameters, after_parameters)
    )
