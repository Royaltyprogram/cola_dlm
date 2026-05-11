import pytest
import torch

from cola_dlm.dit import TimestepEmbedding


def test_timestep_embedding_accepts_vector_and_column_timesteps():
    torch.manual_seed(0)
    embedding = TimestepEmbedding(hidden_size=8)
    timesteps = torch.tensor([0.0, 0.5, 1.0])

    vector_output = embedding(timesteps)
    column_output = embedding(timesteps[:, None])

    assert vector_output.shape == (3, 8)
    assert column_output.shape == (3, 8)
    torch.testing.assert_close(vector_output, column_output)


@pytest.mark.parametrize(
    "timesteps",
    [
        torch.tensor(0.5),
        torch.ones(2, 2),
        torch.ones(2, 1, 1),
    ],
)
def test_timestep_embedding_rejects_invalid_timestep_rank_or_shape(timesteps):
    embedding = TimestepEmbedding(hidden_size=8)

    with pytest.raises(ValueError, match=r"timesteps must be shaped \[batch\]"):
        embedding(timesteps)


def test_timestep_embedding_rejects_non_floating_timesteps():
    embedding = TimestepEmbedding(hidden_size=8)
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="floating point"):
        embedding(timesteps)
