from pathlib import Path

import torch

from cola_dlm.diagnostic_report import (
    build_diagnostics_report,
    render_attention_mask_for_report,
    write_diagnostics_report,
)


def test_build_diagnostics_report_formats_fake_metrics_without_model_run():
    report = build_diagnostics_report(
        stage_name="Stage 2",
        metrics_record={
            "step": 7,
            "loss": 1.25,
            "reconstruction_nll": 2.5,
            "reconstruction_accuracy": 0.75,
            "logsnr": torch.tensor(3.0),
            "latent_norm_mean": 4.0,
            "posterior_variance_mean": 0.5,
            "vae_loss": 1.5,
            "flow_matching_loss": 0.25,
            "reference_kl": 0.125,
        },
        block_loss_metrics={0: torch.tensor(0.2), 1: 0.3},
        attention_mask_text="legend: #=allowed .=denied\nq00 noisy b0: ##",
        checkpoint_path=Path("checkpoints/final.pt"),
        metrics_path=Path("metrics.jsonl"),
    )

    assert report.startswith("# Stage 2 Diagnostics Report\n")
    assert "Final step: 7" in report
    assert "`reconstruction_accuracy`" in report
    assert "`logsnr`" in report
    assert "`posterior_variance_mean`" in report
    assert "`flow_matching_loss_block_0`" in report
    assert "`flow_matching_loss_block_1`" in report
    assert "```text\nlegend: #=allowed .=denied\nq00 noisy b0: ##\n```" in report


def test_write_diagnostics_report_creates_markdown_file(tmp_path):
    path = tmp_path / "diagnostics_report.md"

    returned_path = write_diagnostics_report(
        path,
        stage_name="Stage 1",
        metrics_record={
            "step": 3,
            "reconstruction_nll": 1.0,
            "reconstruction_accuracy": 0.5,
            "kl": 0.1,
            "logsnr": 2.0,
            "latent_norm_mean": 3.0,
            "posterior_variance_mean": 0.9,
        },
    )

    assert returned_path == path
    text = path.read_text(encoding="utf-8")
    assert "Final step: 3" in text
    assert "`reconstruction_nll`" in text
    assert "`kl`" in text


def test_render_attention_mask_for_report_renders_small_masks():
    text = render_attention_mask_for_report(sequence_length=8, block_size=2)

    assert "legend: #=allowed .=denied c=clean n=noisy" in text
    assert "q00 clean b0:" in text


def test_render_attention_mask_for_report_uses_note_for_large_masks():
    text = render_attention_mask_for_report(
        sequence_length=64,
        block_size=4,
        max_packed_length=32,
    )

    assert "Attention mask not rendered" in text
    assert "expected packed length 124" in text
    assert "sequence_length=64" in text
    assert "block_size=4" in text
