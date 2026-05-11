import json
from pathlib import Path

import pytest
import torch
from torch import nn

from cola_dlm.checkpointing import CheckpointError, load_checkpoint, save_checkpoint
from cola_dlm.config import Stage1Config
from cola_dlm.config_io import config_to_dict
from cola_dlm.train_vae_stage1 import main


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_stage1_cli_smoke_writes_checkpoint_config_and_jsonl(tmp_path):
    token_file = _write_token_file(tmp_path)
    output_dir = tmp_path / "run"

    assert main(_args(token_file, output_dir, max_steps=1)) == 0

    checkpoint = output_dir / "checkpoints" / "step_00000001.pt"
    final_checkpoint = output_dir / "checkpoints" / "final.pt"
    config_snapshot = output_dir / "config.json"
    log_file = output_dir / "metrics.jsonl"

    assert checkpoint.exists()
    assert final_checkpoint.exists()
    assert config_snapshot.exists()
    assert log_file.exists()

    loaded = load_checkpoint(final_checkpoint)
    assert loaded.step == 1
    assert loaded.metadata == {"stage": "stage1"}
    assert loaded.config["config"]["vae"]["vocab_size"] == 128

    snapshot = json.loads(config_snapshot.read_text(encoding="utf-8"))
    assert snapshot["config"]["vae"]["sequence_length"] == 16
    assert snapshot["max_steps"] == 1
    assert snapshot["data_files"] == [str(token_file)]

    records = _read_jsonl(log_file)
    assert [record["step"] for record in records] == [1]
    assert {
        "loss",
        "reconstruction_nll",
        "kl",
        "mask_loss",
        "logsnr",
        "reconstruction_accuracy",
        "latent_norm_mean",
        "posterior_variance_mean",
        "lr",
    } <= set(records[0])


def test_stage1_cli_resume_advances_from_saved_step(tmp_path):
    token_file = _write_token_file(tmp_path)
    output_dir = tmp_path / "run"

    main(_args(token_file, output_dir, max_steps=1))
    resume_checkpoint = output_dir / "checkpoints" / "final.pt"
    assert load_checkpoint(resume_checkpoint).step == 1

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


def test_stage1_cli_missing_data_fails_clearly(tmp_path):
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="tokenized text file does not exist"):
        main(_args(missing_file, tmp_path / "run", max_steps=1))


def test_stage1_cli_incompatible_resume_checkpoint_fails_clearly(tmp_path):
    token_file = _write_token_file(tmp_path)
    bad_checkpoint = tmp_path / "bad_resume.pt"
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    save_checkpoint(
        bad_checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        config={"config": config_to_dict(Stage1Config())},
    )

    with pytest.raises(CheckpointError, match="Resume checkpoint is incompatible"):
        main(
            _args(
                token_file,
                tmp_path / "run",
                max_steps=2,
                extra=["--resume", str(bad_checkpoint)],
            )
        )


def test_train_vae_stage1_console_script_is_registered():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert 'train_vae_stage1 = "cola_dlm.train_vae_stage1:main"' in pyproject


def _args(
    token_file: Path,
    output_dir: Path,
    *,
    max_steps: int,
    extra: list[str] | None = None,
) -> list[str]:
    args = [
        "--config",
        str(CONFIG_DIR / "stage1_tiny_debug.json"),
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
