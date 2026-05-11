import importlib
import sys
import types

import pytest

from cola_dlm.tokenizer import (
    HuggingFaceTokenizerAdapter,
    OfflineFallbackTokenizer,
    load_olmo2_tokenizer,
)


def _block_transformers_import(monkeypatch):
    real_import = __import__

    def import_without_transformers(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("No module named transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_without_transformers)


def _install_fake_transformers(monkeypatch, auto_tokenizer):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = auto_tokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_fallback_tokenizer_round_trips_tiny_strings():
    tokenizer = OfflineFallbackTokenizer()

    for text in ("", "a", "hello", "hello\nworld", "cafe"):
        token_ids = tokenizer.encode(text)

        assert tokenizer.encode(text) == token_ids
        assert tokenizer.decode(token_ids) == text


def test_fallback_tokenizer_special_ids_and_exact_byte_mapping():
    tokenizer = OfflineFallbackTokenizer()

    assert tokenizer.pad_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert tokenizer.vocab_size == 258
    assert tokenizer.encode("Az!") == [67, 124, 35]
    assert (
        tokenizer.decode([tokenizer.pad_token_id, 67, tokenizer.eos_token_id])
        == "A"
    )


def test_fallback_tokenizer_rejects_invalid_ids():
    tokenizer = OfflineFallbackTokenizer()

    with pytest.raises(ValueError, match="byte-token range"):
        tokenizer.decode([tokenizer.vocab_size])


def test_load_olmo2_tokenizer_raises_clear_error_without_transformers(monkeypatch):
    _block_transformers_import(monkeypatch)

    with pytest.raises(RuntimeError, match="transformers is not installed"):
        load_olmo2_tokenizer("local-tokenizer", allow_fallback=False)


def test_load_olmo2_tokenizer_can_fall_back_without_transformers(monkeypatch):
    _block_transformers_import(monkeypatch)

    tokenizer = load_olmo2_tokenizer("local-tokenizer", allow_fallback=True)

    assert isinstance(tokenizer, OfflineFallbackTokenizer)


def test_load_olmo2_tokenizer_defaults_to_local_files_only(monkeypatch):
    class FakeAutoTokenizer:
        called_kwargs = None

        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            cls.called_kwargs = kwargs
            raise OSError(f"{model_name_or_path} is not cached")

    _install_fake_transformers(monkeypatch, FakeAutoTokenizer)

    with pytest.raises(RuntimeError, match="local_files_only=True"):
        load_olmo2_tokenizer("local-tokenizer")

    assert FakeAutoTokenizer.called_kwargs == {"local_files_only": True}


def test_load_olmo2_tokenizer_can_fall_back_when_local_files_are_missing(
    monkeypatch,
):
    class FakeAutoTokenizer:
        called_kwargs = None

        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            cls.called_kwargs = kwargs
            raise OSError(f"{model_name_or_path} is not cached")

    _install_fake_transformers(monkeypatch, FakeAutoTokenizer)

    tokenizer = load_olmo2_tokenizer("local-tokenizer", allow_fallback=True)

    assert isinstance(tokenizer, OfflineFallbackTokenizer)
    assert FakeAutoTokenizer.called_kwargs == {"local_files_only": True}


def test_hugging_face_adapter_narrows_tokenizer_interface():
    class FakeTokenizer:
        pad_token_id = 9
        eos_token_id = 10
        vocab_size = 11

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [3, len(text)]

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return f"ids:{','.join(str(token_id) for token_id in token_ids)}"

    tokenizer = HuggingFaceTokenizerAdapter(FakeTokenizer())

    assert tokenizer.pad_token_id == 9
    assert tokenizer.eos_token_id == 10
    assert tokenizer.vocab_size == 11
    assert tokenizer.encode("abc") == [3, 3]
    assert tokenizer.decode([3, 3]) == "ids:3,3"


def test_importing_tokenizer_module_does_not_import_transformers(monkeypatch):
    monkeypatch.delitem(sys.modules, "cola_dlm.tokenizer", raising=False)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)

    module = importlib.import_module("cola_dlm.tokenizer")

    assert module.__all__ == (
        "DEFAULT_OLMO2_TOKENIZER",
        "HuggingFaceTokenizerAdapter",
        "OfflineFallbackTokenizer",
        "TextTokenizer",
        "load_olmo2_tokenizer",
    )
    assert "transformers" not in sys.modules
