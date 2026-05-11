import json
from pathlib import Path

from cola_dlm.checkpointing import save_checkpoint
from cola_dlm.config_io import config_to_dict
from cola_dlm.dit import BlockCausalTextDiT
from cola_dlm.sample_cli import main
from cola_dlm.vae import TextVAE


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_sample_cli_loads_stage2_checkpoint_and_writes_generated_ids(
    tmp_path,
    tiny_stage2_config,
):
    checkpoint = _write_stage2_checkpoint(tmp_path, tiny_stage2_config)
    output = tmp_path / "samples.json"

    assert main(
        [
            "--checkpoint",
            str(checkpoint),
            "--prompt-token-ids",
            "1, 2, 3",
            "--max-new-tokens",
            "2",
            "--output",
            str(output),
            "--device",
            "cpu",
            "--seed",
            "123",
        ]
    ) == 0

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["config_source"] == "checkpoint"
    assert record["checkpoint_step"] == 7
    assert record["prompt_source"] == "token_ids"
    assert record["prompt_token_ids"] == [1, 2, 3]
    assert len(record["generated_token_ids"]) == 2
    assert all(isinstance(token_id, int) for token_id in record["generated_token_ids"])


def test_sample_cli_accepts_direct_prompt_token_ids_with_config_override(
    tmp_path,
    tiny_stage2_config,
):
    checkpoint = _write_stage2_checkpoint(tmp_path, tiny_stage2_config)
    output = tmp_path / "samples.jsonl"

    main(
        [
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(CONFIG_DIR / "inference_tiny_debug.json"),
            "--prompt-token-ids",
            "[4, 5, 6]",
            "--max-new-tokens",
            "1",
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    )

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(records) == 1
    assert records[0]["config_source"] == "override"
    assert records[0]["prompt_token_ids"] == [4, 5, 6]
    assert len(records[0]["generated_token_ids"]) == 1


def test_sample_console_script_is_registered():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert 'sample = "cola_dlm.sample_cli:main"' in pyproject


def _write_stage2_checkpoint(tmp_path: Path, config) -> Path:
    vae = TextVAE(config.vae)
    dit = BlockCausalTextDiT(config.dit)
    checkpoint = tmp_path / "stage2.pt"
    save_checkpoint(
        checkpoint,
        extra_models={"vae": vae, "dit": dit},
        step=7,
        config={"config": config_to_dict(config)},
        metadata={"stage": "stage2"},
    )
    return checkpoint
