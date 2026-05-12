import pytest
import torch

from cola_dlm.transformer import (
    FeedForward,
    MultiHeadAttention,
    OutputProjection,
    RMSNorm,
    RotaryEmbedding,
    TokenEmbedding,
    TransformerBlock,
    TransformerStack,
)


def test_transformer_public_exports_are_core_helpers():
    import cola_dlm.transformer as transformer

    assert transformer.__all__ == (
        "TokenEmbedding",
        "OutputProjection",
        "RMSNorm",
        "FeedForward",
        "RotaryEmbedding",
        "MultiHeadAttention",
        "TransformerBlock",
        "TransformerStack",
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


def test_multi_head_attention_rejects_hidden_size_not_divisible_by_heads():
    with pytest.raises(ValueError, match="hidden_size must be divisible by num_heads"):
        MultiHeadAttention(hidden_size=10, num_heads=3)


def test_multi_head_attention_rejects_mismatched_hidden_size_and_head_dim():
    with pytest.raises(ValueError, match="hidden_size must equal num_heads"):
        MultiHeadAttention(hidden_size=8, num_heads=3, head_dim=4)


def test_multi_head_attention_rejects_odd_rope_head_dim():
    with pytest.raises(ValueError, match="head_dim must be even when use_rope=True"):
        MultiHeadAttention(
            hidden_size=6,
            num_heads=2,
            head_dim=3,
            use_rope=True,
        )


def test_multi_head_attention_causal_mask_blocks_future_tokens():
    torch.manual_seed(0)
    attention = MultiHeadAttention(hidden_size=8, num_heads=2, dropout=0.0)
    attention.eval()
    hidden_states = torch.randn(1, 4, 8)
    changed_future = hidden_states.clone()
    changed_future[:, 2:, :] = torch.randn(1, 2, 8) * 10.0

    output = attention(hidden_states, causal=True)
    changed_output = attention(changed_future, causal=True)

    assert torch.allclose(output[:, :2, :], changed_output[:, :2, :], atol=1.0e-6)


def test_multi_head_attention_boolean_mask_uses_true_for_allowed_positions():
    attention = MultiHeadAttention(hidden_size=2, num_heads=1, dropout=0.0)
    attention.eval()
    with torch.no_grad():
        # Zero query/key scores make each allowed value receive equal attention.
        attention.query_projection.weight.zero_()
        attention.query_projection.bias.zero_()
        attention.key_projection.weight.zero_()
        attention.key_projection.bias.zero_()
        attention.value_projection.weight.copy_(torch.eye(2))
        attention.value_projection.bias.zero_()
        attention.output_projection.weight.copy_(torch.eye(2))
        attention.output_projection.bias.zero_()

    hidden_states = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]])
    allowed_positions = torch.tensor(
        [
            [True, False, False],
            [False, True, True],
            [True, False, True],
        ]
    )

    output = attention(hidden_states, attention_mask=allowed_positions)

    expected = torch.tensor([[[1.0, 0.0], [1.5, 3.0], [2.0, 2.0]]])
    assert torch.allclose(output, expected)


def test_multi_head_attention_accepts_boolean_batch_key_padding_mask():
    attention = MultiHeadAttention(hidden_size=2, num_heads=1, dropout=0.0)
    attention.eval()
    with torch.no_grad():
        # Zero query/key scores make each allowed value receive equal attention.
        attention.query_projection.weight.zero_()
        attention.query_projection.bias.zero_()
        attention.key_projection.weight.zero_()
        attention.key_projection.bias.zero_()
        attention.value_projection.weight.copy_(torch.eye(2))
        attention.value_projection.bias.zero_()
        attention.output_projection.weight.copy_(torch.eye(2))
        attention.output_projection.bias.zero_()

    hidden_states = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]],
            [[10.0, 0.0], [0.0, 20.0], [30.0, 40.0]],
        ]
    )
    key_padding_mask = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
        ]
    )

    output = attention(hidden_states, attention_mask=key_padding_mask)

    expected = torch.tensor(
        [
            [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
            [[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]],
        ]
    )
    assert torch.allclose(output, expected)


def test_multi_head_attention_accepts_additive_attention_masks():
    attention = MultiHeadAttention(hidden_size=8, num_heads=2, dropout=0.0)
    hidden_states = torch.randn(2, 4, 8)
    attention_mask = torch.zeros(2, 2, 4, 4)
    attention_mask[:, :, :, -1] = -100.0

    output = attention(hidden_states, attention_mask=attention_mask)

    assert output.shape == (2, 4, 8)


def test_multi_head_attention_rejects_unsupported_attention_mask_shape():
    attention = MultiHeadAttention(hidden_size=2, num_heads=1, dropout=0.0)
    hidden_states = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    one_dimensional_mask = torch.ones(3, dtype=torch.bool)

    with pytest.raises(ValueError, match="attention_mask must be shaped"):
        attention(hidden_states, attention_mask=one_dimensional_mask)


def test_transformer_block_preserves_hidden_shape_with_mask_and_causal_flag():
    block = TransformerBlock(
        hidden_size=8,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
    )
    hidden_states = torch.randn(2, 4, 8)
    attention_mask = torch.ones(4, 4, dtype=torch.bool)

    output = block(hidden_states, attention_mask=attention_mask, causal=True)

    assert output.shape == hidden_states.shape


def test_tiny_transformer_stack_forward_produces_logits_shape_in_eval_mode():
    torch.manual_seed(0)
    vocab_size = 13
    hidden_size = 8
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    attention_mask = torch.ones(token_ids.shape[1], token_ids.shape[1], dtype=torch.bool)

    embedding = TokenEmbedding(vocab_size=vocab_size, hidden_size=hidden_size)
    transformer = TransformerStack(
        num_layers=2,
        hidden_size=hidden_size,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
    )
    projection = OutputProjection(hidden_size=hidden_size, vocab_size=vocab_size)
    embedding.eval()
    transformer.eval()
    projection.eval()

    with torch.no_grad():
        hidden_states = embedding(token_ids)
        hidden_states = transformer(
            hidden_states,
            attention_mask=attention_mask,
            causal=True,
        )
        logits = projection(hidden_states)
        repeat_logits = projection(
            transformer(
                embedding(token_ids),
                attention_mask=attention_mask,
                causal=True,
            )
        )

    assert logits.shape == (2, 4, vocab_size)
    assert torch.allclose(logits, repeat_logits)
