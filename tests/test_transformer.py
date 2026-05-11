import pytest
import torch

from cola_dlm.transformer import (
    FeedForward,
    OutputProjection,
    RMSNorm,
    RotaryEmbedding,
    TokenEmbedding,
)


def test_transformer_public_exports_are_core_helpers():
    import cola_dlm.transformer as transformer

    assert transformer.__all__ == (
        "TokenEmbedding",
        "OutputProjection",
        "RMSNorm",
        "FeedForward",
        "RotaryEmbedding",
    )


def test_token_embedding_maps_token_ids_to_hidden_states():
    embedding = TokenEmbedding(vocab_size=16, hidden_size=8)
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    hidden_states = embedding(token_ids)

    assert hidden_states.shape == (2, 3, 8)


def test_output_projection_maps_hidden_states_to_logits():
    projection = OutputProjection(hidden_size=8, vocab_size=16)
    hidden_states = torch.randn(2, 3, 8)

    logits = projection(hidden_states)

    assert logits.shape == (2, 3, 16)


def test_rms_norm_preserves_shape_and_matches_local_formula():
    norm = RMSNorm(hidden_size=3)
    hidden_states = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 0.0, 4.0]]])

    normalized = norm(hidden_states)

    expected = hidden_states * torch.rsqrt(
        hidden_states.pow(2).mean(dim=-1, keepdim=True) + norm.eps
    )
    assert normalized.shape == hidden_states.shape
    assert torch.allclose(normalized, expected)


@pytest.mark.parametrize("activation", ["gelu", "silu"])
def test_feed_forward_preserves_hidden_shape(activation):
    feed_forward = FeedForward(
        hidden_size=8,
        ffn_size=16,
        dropout=0.0,
        activation=activation,
    )
    hidden_states = torch.randn(2, 3, 8)

    output = feed_forward(hidden_states)

    assert output.shape == (2, 3, 8)


def test_feed_forward_rejects_unknown_activation():
    with pytest.raises(ValueError, match="activation must be 'gelu' or 'silu'"):
        FeedForward(hidden_size=8, ffn_size=16, activation="relu")


def test_rotary_embedding_preserves_query_key_shapes_and_dtype():
    rotary = RotaryEmbedding()
    query = torch.randn(2, 3, 4, 6, dtype=torch.float64)
    key = torch.randn(2, 3, 4, 6, dtype=torch.float64)

    rotated_query, rotated_key = rotary(query, key)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    assert rotated_query.dtype == query.dtype
    assert rotated_key.dtype == key.dtype


def test_rotary_embedding_rejects_odd_head_dim():
    rotary = RotaryEmbedding()
    query = torch.randn(1, 2, 3, 5)
    key = torch.randn(1, 2, 3, 5)

    with pytest.raises(ValueError, match="head_dim must be even"):
        rotary(query, key)


def test_rotary_embedding_keeps_first_position_unchanged():
    rotary = RotaryEmbedding()
    query = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]]])
    key = torch.tensor([[[[4.0, 3.0, 2.0, 1.0], [8.0, 6.0, 4.0, 2.0]]]])

    rotated_query, rotated_key = rotary(query, key)

    assert torch.allclose(rotated_query[:, :, 0, :], query[:, :, 0, :])
    assert torch.allclose(rotated_key[:, :, 0, :], key[:, :, 0, :])
