from pathlib import Path

import pytest
from torch import nn

from cola_dlm.dit import BlockCausalTextDiT
from cola_dlm.parameter_count import (
    PAPER_DIT_NON_EMBEDDING_PARAMETERS,
    PAPER_VAE_PARAMETERS,
    count_dit_components,
    count_embedding_parameters,
    count_non_embedding_backbone_parameters,
    count_parameters,
    count_vae_components,
    main,
    meta_device_is_available,
)
from cola_dlm.vae import TextVAE


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
PAPER_SCALE_DOC = REPO_ROOT / "docs/reproduction/cola_dlm/paper_scale_config.md"
PARAMETER_REPORT = REPO_ROOT / "docs/reproduction/cola_dlm/parameter_counts.md"


class TinyKnownParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(5, 3)
        self.linear = nn.Linear(3, 2)
        self.frozen = nn.Linear(2, 2, bias=False)
        self.frozen.requires_grad_(False)


def test_parameter_helpers_count_known_tiny_model():
    model = TinyKnownParameterModel()

    counts = count_parameters(model)

    assert counts.trainable == 23
    assert counts.non_trainable == 4
    assert counts.total == 27
    assert count_embedding_parameters(model) == 15
    assert count_non_embedding_backbone_parameters(model) == 12


def test_tiny_vae_component_counts_include_output_projection(tiny_vae_config):
    model = TextVAE(tiny_vae_config)

    counts = count_vae_components(model)

    expected_vocab_matrix = tiny_vae_config.vocab_size * tiny_vae_config.hidden_size
    assert counts.encoder.total + counts.decoder.total == counts.total.total
    assert counts.encoder_embedding_parameters == expected_vocab_matrix
    assert counts.decoder_embedding_parameters == expected_vocab_matrix
    assert counts.embedding_parameters == 2 * expected_vocab_matrix
    assert counts.decoder_output_projection_parameters == expected_vocab_matrix


def test_tiny_dit_component_counts_partition_total(tiny_dit_config):
    model = BlockCausalTextDiT(tiny_dit_config)

    counts = count_dit_components(model)
    component_total = (
        counts.input_projection_parameters
        + counts.timestep_embedding_parameters
        + counts.transformer_layer_parameters
        + counts.segment_embedding_parameters
        + counts.output_head_parameters
    )

    assert counts.embedding_parameters == 0
    assert counts.non_embedding_backbone_parameters == counts.total.total
    assert component_total == counts.total.total


def test_paper_report_command_uses_meta_and_writes_markdown(tmp_path):
    if not meta_device_is_available():
        pytest.skip("PyTorch meta-device module initialization is unavailable")
    output = tmp_path / "parameter_counts.md"

    exit_code = main(
        [
            "--config",
            str(CONFIG_DIR / "stage2_paper.json"),
            "--stage1-config",
            str(CONFIG_DIR / "stage1_paper.json"),
            "--output",
            str(output),
        ],
    )

    text = output.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "about 500M" in text
    assert "about 1.8B" in text
    assert "configs/stage2_paper.json" in text
    assert "VAE initialization device: `meta`" in text
    assert "DiT initialization device: `meta`" in text


def test_paper_scale_docs_and_report_use_same_paths_and_paper_values():
    report_text = PARAMETER_REPORT.read_text(encoding="utf-8").replace("`", "")
    doc_text = PAPER_SCALE_DOC.read_text(encoding="utf-8").replace("`", "")

    for stable_path in ("configs/stage1_paper.json", "configs/stage2_paper.json"):
        assert stable_path in report_text
        assert stable_path in doc_text

    for paper_value in (
        PAPER_VAE_PARAMETERS,
        PAPER_DIT_NON_EMBEDDING_PARAMETERS,
    ):
        assert paper_value in report_text
        assert paper_value in doc_text
