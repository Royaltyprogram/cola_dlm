from dataclasses import replace

import pytest
import torch

from cola_dlm.config import Stage2Config
from cola_dlm.dit import BlockCausalTextDiT
from cola_dlm.stage2 import create_frozen_reference_encoder
from cola_dlm.vae import TextVAE, TextVAEEncoder


def test_stage2_public_surface_starts_with_reference_encoder_helper():
    import cola_dlm.stage2 as stage2

    assert stage2.__all__ == ("create_frozen_reference_encoder",)
    assert stage2.create_frozen_reference_encoder is create_frozen_reference_encoder


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
