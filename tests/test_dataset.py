import pytest
import torch

from cola_dlm.dataset import TokenizedTextDataset


def test_tokenized_text_dataset_public_surface():
    import cola_dlm.dataset as dataset

    assert dataset.__all__ == ("TokenizedTextDataset",)


def test_tokenized_text_dataset_returns_exact_non_overlapping_chunks(tmp_path):
    token_file = _write_tokens(tmp_path, "tokens.txt", "1 2 3 4\n5 6 7 8\n")

    dataset = TokenizedTextDataset(token_file, sequence_length=4)

    assert len(dataset) == 2
    _assert_sample(
        dataset[0],
        input_ids=[1, 2, 3, 4],
        attention_mask=[1, 1, 1, 1],
    )
    _assert_sample(
        dataset[1],
        input_ids=[5, 6, 7, 8],
        attention_mask=[1, 1, 1, 1],
    )


def test_tokenized_text_dataset_pads_final_short_chunk_across_files(tmp_path):
    first_file = _write_tokens(tmp_path, "first.txt", "10 11\n")
    second_file = _write_tokens(tmp_path, "second.txt", "12 13 14\n")

    dataset = TokenizedTextDataset(
        [first_file, second_file],
        sequence_length=4,
        pad_token_id=99,
    )

    assert len(dataset) == 2
    _assert_sample(
        dataset[0],
        input_ids=[10, 11, 12, 13],
        attention_mask=[1, 1, 1, 1],
    )
    _assert_sample(
        dataset[1],
        input_ids=[14, 99, 99, 99],
        attention_mask=[1, 0, 0, 0],
    )


def test_tokenized_text_dataset_drops_final_short_chunk(tmp_path):
    token_file = _write_tokens(tmp_path, "tokens.txt", "1 2 3 4 5")

    dataset = TokenizedTextDataset(
        token_file,
        sequence_length=4,
        drop_last=True,
    )

    assert len(dataset) == 1
    _assert_sample(
        dataset[0],
        input_ids=[1, 2, 3, 4],
        attention_mask=[1, 1, 1, 1],
    )


def test_tokenized_text_dataset_uses_configured_stride(tmp_path):
    token_file = _write_tokens(tmp_path, "tokens.txt", "1 2 3 4 5 6 7")

    dataset = TokenizedTextDataset(
        token_file,
        sequence_length=4,
        stride=2,
        pad_token_id=0,
    )

    assert len(dataset) == 3
    _assert_sample(
        dataset[0],
        input_ids=[1, 2, 3, 4],
        attention_mask=[1, 1, 1, 1],
    )
    _assert_sample(
        dataset[1],
        input_ids=[3, 4, 5, 6],
        attention_mask=[1, 1, 1, 1],
    )
    _assert_sample(
        dataset[2],
        input_ids=[5, 6, 7, 0],
        attention_mask=[1, 1, 1, 0],
    )


def test_tokenized_text_dataset_rejects_malformed_token_ids(tmp_path):
    token_file = _write_tokens(tmp_path, "tokens.txt", "1 2\n3 nope\n")

    with pytest.raises(ValueError, match="malformed token id 'nope'.*line 2"):
        TokenizedTextDataset(token_file, sequence_length=4)


def test_tokenized_text_dataset_rejects_missing_files(tmp_path):
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        TokenizedTextDataset(missing_file, sequence_length=4)


def test_tokenized_text_dataset_rejects_invalid_chunk_configuration(tmp_path):
    token_file = _write_tokens(tmp_path, "tokens.txt", "1 2 3 4")

    with pytest.raises(ValueError, match="sequence_length must be a positive"):
        TokenizedTextDataset(token_file, sequence_length=0)

    with pytest.raises(ValueError, match="stride must be a positive"):
        TokenizedTextDataset(token_file, sequence_length=4, stride=0)

    with pytest.raises(ValueError, match="stride must be no larger"):
        TokenizedTextDataset(token_file, sequence_length=4, stride=5)

    with pytest.raises(ValueError, match="pad_token_id must be a non-negative"):
        TokenizedTextDataset(token_file, sequence_length=4, pad_token_id=-1)


def _write_tokens(tmp_path, name: str, text: str):
    token_file = tmp_path / name
    token_file.write_text(text, encoding="utf-8")
    return token_file


def _assert_sample(
    sample: dict[str, torch.Tensor],
    *,
    input_ids: list[int],
    attention_mask: list[int],
) -> None:
    assert tuple(sample) == ("input_ids", "attention_mask")
    assert torch.equal(sample["input_ids"], torch.tensor(input_ids, dtype=torch.long))
    assert torch.equal(
        sample["attention_mask"],
        torch.tensor(attention_mask, dtype=torch.bool),
    )
