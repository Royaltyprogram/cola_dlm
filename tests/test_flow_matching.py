import pytest
import torch
import torch.nn.functional as F

import cola_dlm.flow_matching as flow_matching
from cola_dlm.config import DiffusionConfig
from cola_dlm.flow_matching import (
    flow_matching_loss,
    flow_matching_target,
    linear_bridge,
    sample_timestep,
    velocity_target,
    x0_target,
)


def test_flow_matching_public_surface_is_concise():
    assert flow_matching.__all__ == (
        "flow_matching_loss",
        "flow_matching_target",
        "linear_bridge",
        "sample_timestep",
        "velocity_target",
        "x0_target",
    )


def test_uniform_timestep_samples_have_requested_shape_and_open_range():
    config = DiffusionConfig(timestep_schedule="uniform")

    timesteps = sample_timestep(config, (2, 3), dtype=torch.float64)

    assert timesteps.shape == (2, 3)
    assert timesteps.dtype is torch.float64
    assert torch.all(timesteps > 0)
    assert torch.all(timesteps < 1)


def test_logit_normal_timestep_uses_configured_loc_and_scale():
    config = DiffusionConfig(
        timestep_schedule="logit_normal",
        logit_normal_loc=-0.5,
        logit_normal_scale=0.25,
    )
    generator = torch.Generator().manual_seed(11)
    expected_generator = torch.Generator().manual_seed(11)

    timesteps = sample_timestep(config, 4, generator=generator)
    expected = torch.sigmoid(
        torch.randn(4, generator=expected_generator) * 0.25 - 0.5
    )

    torch.testing.assert_close(timesteps, expected)


def test_logit_normal_timestep_defaults_missing_scale_to_one():
    config = DiffusionConfig(
        timestep_schedule="logit_normal",
        logit_normal_loc=0.75,
        logit_normal_scale=None,
    )
    generator = torch.Generator().manual_seed(29)
    expected_generator = torch.Generator().manual_seed(29)

    timesteps = sample_timestep(config, 3, generator=generator)
    expected = torch.sigmoid(torch.randn(3, generator=expected_generator) + 0.75)

    torch.testing.assert_close(timesteps, expected)


def test_discrete_timestep_samples_belong_to_normalized_midpoint_grid():
    config = DiffusionConfig(timestep_schedule="uniform")
    generator = torch.Generator().manual_seed(37)

    timesteps = sample_timestep(
        config,
        (4, 5),
        dtype=torch.float64,
        generator=generator,
        num_discrete_timesteps=4,
    )

    grid = torch.tensor([0.125, 0.375, 0.625, 0.875], dtype=torch.float64)
    belongs_to_grid = (timesteps[..., None] == grid).any(dim=-1)
    assert torch.all(belongs_to_grid)


def test_seeded_generators_make_timestep_sampling_deterministic():
    config = DiffusionConfig(timestep_schedule="uniform")

    first = sample_timestep(
        config,
        batch_size=6,
        generator=torch.Generator().manual_seed(41),
    )
    second = sample_timestep(
        config,
        batch_size=6,
        generator=torch.Generator().manual_seed(41),
    )

    torch.testing.assert_close(first, second)


def test_sample_timestep_rejects_invalid_inputs():
    config = DiffusionConfig(timestep_schedule="uniform")

    with pytest.raises(TypeError, match="floating point"):
        sample_timestep(config, 2, dtype=torch.long)

    with pytest.raises(ValueError, match="num_discrete_timesteps must be positive"):
        sample_timestep(config, 2, num_discrete_timesteps=0)

    with pytest.raises(TypeError, match="num_discrete_timesteps"):
        sample_timestep(config, 2, num_discrete_timesteps=2.5)

    config.timestep_schedule = "unknown"
    with pytest.raises(ValueError, match="timestep_schedule"):
        sample_timestep(config, 2)

    config = DiffusionConfig(timestep_schedule="logit_normal", logit_normal_scale=0.0)
    with pytest.raises(ValueError, match="logit_normal_scale must be positive"):
        sample_timestep(config, 2)


def test_linear_bridge_preserves_latent_shape():
    z0 = torch.zeros(2, 3, 4)
    z1 = torch.ones(2, 3, 4)
    timesteps = torch.tensor([0.25, 0.75])

    bridged = linear_bridge(z0, z1, timesteps)

    assert bridged.shape == z0.shape


def test_linear_bridge_matches_endpoints():
    z0 = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    z1 = torch.tensor(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[50.0, 60.0], [70.0, 80.0]],
        ]
    )

    torch.testing.assert_close(linear_bridge(z0, z1, torch.zeros(2)), z0)
    torch.testing.assert_close(linear_bridge(z0, z1, torch.ones(2, 1)), z1)


def test_linear_bridge_matches_midpoint_formula():
    z0 = torch.tensor([[[0.0, 2.0], [4.0, 6.0]]])
    z1 = torch.tensor([[[2.0, 4.0], [8.0, 10.0]]])

    bridged = linear_bridge(z0, z1, torch.tensor([0.5]))

    expected = torch.tensor([[[1.0, 3.0], [6.0, 8.0]]])
    torch.testing.assert_close(bridged, expected)


def test_velocity_target_matches_analytical_linear_bridge_formula():
    z0 = torch.tensor([[[1.0, -1.0], [3.0, 4.0]]])
    z1 = torch.tensor([[[2.5, 0.0], [-1.0, 10.0]]])
    timestep = torch.tensor([0.25])

    target = velocity_target(z0, z1)

    broadcast_timestep = timestep.reshape(1, 1, 1)
    expected_z_t = (1 - broadcast_timestep) * z0 + broadcast_timestep * z1
    expected_velocity = z1 - z0
    torch.testing.assert_close(linear_bridge(z0, z1, timestep), expected_z_t)
    torch.testing.assert_close(target, expected_velocity)


def test_flow_matching_target_selects_velocity_target():
    z0 = torch.tensor([[[1.0, 2.0]]])
    z1 = torch.tensor([[[4.0, 6.0]]])

    target = flow_matching_target(z0, z1, "velocity")

    torch.testing.assert_close(target, z1 - z0)


def test_x0_prediction_type_returns_clean_latent_target():
    z0 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    z1 = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])

    torch.testing.assert_close(x0_target(z0), z0)
    torch.testing.assert_close(flow_matching_target(z0, z1, "x0"), z0)


def test_flow_matching_target_rejects_unsupported_prediction_type():
    z0 = torch.zeros(1, 2, 3)
    z1 = torch.ones(1, 2, 3)

    with pytest.raises(ValueError, match="prediction_type"):
        flow_matching_target(z0, z1, "epsilon")


def test_flow_matching_velocity_utility_path_computes_finite_loss():
    config = DiffusionConfig(prediction_type="velocity", timestep_schedule="uniform")
    z0 = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, -2.0], [-3.0, -4.0]],
        ]
    )
    z1 = torch.tensor(
        [
            [[2.0, 4.0], [6.0, 8.0]],
            [[0.0, 1.0], [2.0, 3.0]],
        ]
    )
    timestep = sample_timestep(
        config,
        batch_size=z0.shape[0],
        dtype=z0.dtype,
        generator=torch.Generator().manual_seed(53),
    )

    z_t = linear_bridge(z0, z1, timestep)
    target = flow_matching_target(z0, z1, config.prediction_type)
    prediction = torch.zeros_like(z_t)
    loss = flow_matching_loss(prediction, target)

    assert z_t.shape == z0.shape
    assert target.shape == z0.shape
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss).item()


def test_flow_matching_loss_matches_unmasked_mse_loss():
    prediction = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    target = torch.tensor([[[0.5, 1.5], [2.0, 6.0]]])

    loss = flow_matching_loss(prediction, target)

    torch.testing.assert_close(loss, F.mse_loss(prediction, target))


def test_x0_prediction_type_computes_loss_against_clean_latents():
    config = DiffusionConfig(prediction_type="x0")
    z0 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    z1 = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    prediction = torch.tensor([[[1.5, 2.5], [2.0, 6.0]]])

    target = flow_matching_target(z0, z1, config.prediction_type)
    loss = flow_matching_loss(prediction, target)

    torch.testing.assert_close(target, z0)
    torch.testing.assert_close(loss, F.mse_loss(prediction, z0))


def test_flow_matching_loss_broadcasts_position_mask_across_latents():
    prediction = torch.tensor([[[1.0, 2.0], [10.0, 20.0], [3.0, 4.0]]])
    target = torch.zeros_like(prediction)
    loss_mask = torch.tensor([[True, False, True]])

    loss = flow_matching_loss(prediction, target, loss_mask)

    expected = torch.tensor((1.0 + 4.0 + 9.0 + 16.0) / 4.0)
    torch.testing.assert_close(loss, expected)


def test_flow_matching_loss_accepts_prediction_shaped_float_mask():
    prediction = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    target = torch.zeros_like(prediction)
    loss_mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    loss = flow_matching_loss(prediction, target, loss_mask)

    expected = torch.tensor((1.0 + 16.0) / 2.0)
    torch.testing.assert_close(loss, expected)


def test_flow_matching_loss_returns_finite_zero_for_empty_mask():
    prediction = torch.ones(1, 2, 3, requires_grad=True)
    target = torch.zeros_like(prediction)
    loss_mask = torch.zeros(1, 2, dtype=torch.bool)

    loss = flow_matching_loss(prediction, target, loss_mask)

    assert loss.shape == torch.Size([])
    assert loss.device == prediction.device
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_flow_matching_loss_rejects_shape_mismatches():
    prediction = torch.zeros(1, 2, 3)
    target = torch.zeros(1, 2, 4)

    with pytest.raises(ValueError, match="prediction and target"):
        flow_matching_loss(prediction, target)

    with pytest.raises(ValueError, match="loss_mask"):
        flow_matching_loss(prediction, torch.zeros_like(prediction), torch.ones(2, 2))


def test_flow_matching_loss_requires_floating_prediction_and_target():
    prediction = torch.zeros(1, 2, 3)

    with pytest.raises(TypeError, match="target"):
        flow_matching_loss(prediction, torch.zeros(1, 2, 3, dtype=torch.long))

    with pytest.raises(TypeError, match="loss_mask"):
        flow_matching_loss(prediction, prediction, torch.ones(1, 2, dtype=torch.long))
