import json
from pathlib import Path

import torch

from cola_dlm.checkpointing import load_checkpoint
from cola_dlm.config import Stage2Config
from cola_dlm.config_io import load_config
from cola_dlm.dit import BlockCausalTextDiT
from cola_dlm.train_stage2 import _build_optimizer, _load_resume_checkpoint, main
from cola_dlm.training_utils import build_scheduler
from cola_dlm.vae import TextVAE, TextVAEEncoder


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_stage2_cli_smoke_writes_named_checkpoint_and_jsonl(tmp_path):
    token_file = _write_token_file(tmp_path)
    output_dir = tmp_path / "run"

    assert main(_args(token_file, output_dir, max_steps=1)) == 0

    checkpoint = output_dir / "checkpoints" / "step_00000001.pt"
    final_checkpoint = output_dir / "checkpoints" / "final.pt"
    config_snapshot = output_dir / "config.json"
    log_file = output_dir / "metrics.jsonl"
    diagnostics_report = output_dir / "diagnostics_report.md"

    assert checkpoint.exists()
    assert final_checkpoint.exists()
    assert config_snapshot.exists()
    assert log_file.exists()
    assert diagnostics_report.exists()

    payload = _load_raw_checkpoint(final_checkpoint)
    assert payload["model"] is None
    assert set(payload["extra_models"]) == {"vae", "dit"}

    loaded = load_checkpoint(final_checkpoint)
    assert loaded.step == 1
    assert loaded.metadata == {"stage": "stage2"}
    assert loaded.config["config"]["vae"]["vocab_size"] == 128

    snapshot = json.loads(config_snapshot.read_text(encoding="utf-8"))
    assert snapshot["config"]["vae"]["sequence_length"] == 16
    assert snapshot["config"]["dit"]["block_size"] == 4
    assert snapshot["max_steps"] == 1
    assert snapshot["data_files"] == [str(token_file)]

    records = _read_jsonl(log_file)
    assert [record["step"] for record in records] == [1]
    assert {
        "loss",
        "vae_loss",
        "flow_matching_loss",
        "flow_matching_loss_block_0",
        "reference_kl",
        "reconstruction_accuracy",
        "logsnr",
        "latent_norm_mean",
        "posterior_variance_mean",
    } <= set(records[0])

    report = diagnostics_report.read_text(encoding="utf-8")
    assert "Final step: 1" in report
    assert "`reconstruction_nll`" in report
    assert "`reconstruction_accuracy`" in report
    assert "`logsnr`" in report
    assert "`latent_norm_mean`" in report
    assert "`posterior_variance_mean`" in report
    assert "`vae_loss`" in report
    assert "`flow_matching_loss`" in report
    assert "`reference_kl`" in report
    assert "`flow_matching_loss_block_0`" in report
    assert "legend: #=allowed .=denied c=clean n=noisy" in report
    assert "q00 clean b0:" in report


def test_stage2_cli_resume_restores_states_advances_and_keeps_reference_frozen(
    tmp_path,
):
    token_file = _write_token_file(tmp_path)
    output_dir = tmp_path / "run"
    config = load_config(CONFIG_DIR / "stage2_tiny_debug.json", Stage2Config).config

    main(_args(token_file, output_dir, max_steps=1))
    resume_checkpoint = output_dir / "checkpoints" / "final.pt"

    expected_vae = TextVAE(config.vae)
    expected_dit = BlockCausalTextDiT(config.dit)
    load_checkpoint(
        resume_checkpoint,
        extra_models={"vae": expected_vae, "dit": expected_dit},
    )

    resumed_vae = TextVAE(config.vae)
    resumed_dit = BlockCausalTextDiT(config.dit)
    optimizer = _build_optimizer(
        resumed_vae,
        resumed_dit,
        config.optimizer,
        config.vae_dit_lr_ratio,
    )
    scheduler = build_scheduler(optimizer, config.optimizer, max_steps=2)
    step, reference_encoder = _load_resume_checkpoint(
        resume_checkpoint,
        vae=resumed_vae,
        dit=resumed_dit,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=torch.device("cpu"),
    )

    assert step == 1
    _assert_same_state_dict(expected_vae, resumed_vae)
    _assert_same_state_dict(expected_dit, resumed_dit)
    _assert_reference_frozen(reference_encoder)

    main(
        _args(
            token_file,
            output_dir,
            max_steps=2,
            extra=["--resume", str(resume_checkpoint)],
        )
    )

    final_checkpoint = output_dir / "checkpoints" / "final.pt"
    assert load_checkpoint(final_checkpoint).step == 2
    records = _read_jsonl(output_dir / "metrics.jsonl")
    assert [record["step"] for record in records] == [1, 2]

    after_vae = TextVAE(config.vae)
    after_dit = BlockCausalTextDiT(config.dit)
    after_optimizer = _build_optimizer(
        after_vae,
        after_dit,
        config.optimizer,
        config.vae_dit_lr_ratio,
    )
    after_scheduler = build_scheduler(after_optimizer, config.optimizer, max_steps=2)
    after_step, after_reference_encoder = _load_resume_checkpoint(
        final_checkpoint,
        vae=after_vae,
        dit=after_dit,
        optimizer=after_optimizer,
        scheduler=after_scheduler,
        config=config,
        device=torch.device("cpu"),
    )

    assert after_step == 2
    _assert_reference_frozen(after_reference_encoder)


def test_train_stage2_console_script_is_registered():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert 'train_stage2 = "cola_dlm.train_stage2:main"' in pyproject


def _args(
    token_file: Path,
    output_dir: Path,
    *,
    max_steps: int,
    extra: list[str] | None = None,
) -> list[str]:
    args = [
        "--config",
        str(CONFIG_DIR / "stage2_tiny_debug.json"),
        "--data",
        str(token_file),
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(max_steps),
        "--batch-size",
        "2",
        "--checkpoint-every",
        "1",
        "--log-every",
        "1",
        "--device",
        "cpu",
        "--seed",
        "0",
    ]
    if extra is not None:
        args.extend(extra)
    return args


def _write_token_file(tmp_path: Path) -> Path:
    token_file = tmp_path / "tokens.txt"
    token_ids = [str(index % 64) for index in range(64)]
    token_file.write_text(" ".join(token_ids), encoding="utf-8")
    return token_file


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _load_raw_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _assert_same_state_dict(
    expected_module: torch.nn.Module,
    actual_module: torch.nn.Module,
) -> None:
    expected = expected_module.state_dict()
    actual = actual_module.state_dict()
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


def _assert_reference_frozen(reference_encoder: TextVAEEncoder) -> None:
    assert reference_encoder.training is False
    assert all(
        not parameter.requires_grad for parameter in reference_encoder.parameters()
    )
