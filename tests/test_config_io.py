import json
from pathlib import Path

import pytest

from cola_dlm.config import InferenceConfig, Stage1Config, Stage2Config
from cola_dlm.config_io import (
    LoadedConfig,
    config_from_dict,
    config_to_dict,
    load_config,
    save_config,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONTEXT_DOC = "docs/reproduction/cola_dlm/00_context.md"


def _write_recipe(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_stage1_config_dict_round_trip(tiny_stage1_config):
    values = config_to_dict(tiny_stage1_config)

    restored = config_from_dict(Stage1Config, values)

    assert restored == tiny_stage1_config
    assert values["optimizer"]["betas"] == [0.9, 0.95]


def test_stage2_config_dict_round_trip(tiny_stage2_config):
    values = config_to_dict(tiny_stage2_config)

    restored = config_from_dict(Stage2Config, values)

    assert restored == tiny_stage2_config


def test_inference_config_dict_round_trip(tiny_inference_config):
    values = config_to_dict(tiny_inference_config)

    restored = config_from_dict(InferenceConfig, values)

    assert restored == tiny_inference_config


def test_save_and_load_config_preserves_metadata(tmp_path, tiny_stage1_config):
    path = tmp_path / "stage1.json"

    save_config(
        tiny_stage1_config,
        path,
        metadata={"max_steps": 2, "output_dir": "runs/tiny"},
    )
    loaded = load_config(path, Stage1Config)

    assert loaded == LoadedConfig(
        config=tiny_stage1_config,
        metadata={"max_steps": 2, "output_dir": "runs/tiny"},
    )


def test_inherited_recipe_without_overrides_matches_base_config(
    tmp_path,
    tiny_stage2_config,
):
    base_path = tmp_path / "base.json"
    child_path = tmp_path / "child.json"
    _write_recipe(
        base_path,
        {
            "description": "base",
            "config": config_to_dict(tiny_stage2_config),
        },
    )
    _write_recipe(
        child_path,
        {
            "extends": "base.json",
            "description": "child",
        },
    )

    loaded = load_config(child_path, Stage2Config)

    assert loaded.config == tiny_stage2_config
    assert loaded.metadata == {"description": "child"}


def test_inherited_recipe_merges_config_metadata_and_relative_extends(
    tmp_path,
    tiny_stage1_config,
):
    base_path = tmp_path / "paper" / "base.json"
    child_path = tmp_path / "variants" / "child.json"
    _write_recipe(
        base_path,
        {
            "description": "base",
            "source_document": CONTEXT_DOC,
            "owner": "base",
            "config": config_to_dict(tiny_stage1_config),
        },
    )
    _write_recipe(
        child_path,
        {
            "extends": "../paper/base.json",
            "description": "child",
            "owner": "child",
            "config": {
                "optimizer": {
                    "betas": [0.8, 0.9],
                    "peak_lr": 2.0e-4,
                },
                "global_batch_size": 4,
            },
        },
    )

    loaded = load_config(child_path, Stage1Config)

    assert loaded.config.optimizer.peak_lr == 2.0e-4
    assert loaded.config.optimizer.betas == (0.8, 0.9)
    assert (
        loaded.config.optimizer.warmup_steps
        == tiny_stage1_config.optimizer.warmup_steps
    )
    assert loaded.config.global_batch_size == 4
    assert loaded.config.tokens_per_step == tiny_stage1_config.tokens_per_step
    assert loaded.metadata == {
        "description": "child",
        "source_document": CONTEXT_DOC,
        "owner": "child",
    }


def test_inherited_recipe_missing_base_fails_clearly(tmp_path):
    child_path = tmp_path / "child.json"
    _write_recipe(child_path, {"extends": "missing.json"})

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(child_path, Stage1Config)


def test_inherited_recipe_requires_string_extends(tmp_path):
    child_path = tmp_path / "child.json"
    _write_recipe(child_path, {"extends": None})

    with pytest.raises(TypeError, match="extends must be a string"):
        load_config(child_path, Stage1Config)


def test_inherited_recipe_cycle_fails_clearly(tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "nested" / "second.json"
    _write_recipe(first_path, {"extends": "nested/second.json"})
    _write_recipe(second_path, {"extends": "../first.json"})

    with pytest.raises(ValueError, match="Config inheritance cycle detected"):
        load_config(first_path, Stage1Config)


def test_recipe_files_load_without_optional_dependencies():
    recipes = [
        ("stage1_tiny_debug.json", Stage1Config),
        ("stage2_tiny_debug.json", Stage2Config),
        ("inference_tiny_debug.json", InferenceConfig),
        ("stage1_paper.json", Stage1Config),
        ("stage2_paper.json", Stage2Config),
        ("paper/stage2_paper_base.json", Stage2Config),
    ]

    for file_name, config_type in recipes:
        loaded = load_config(CONFIG_DIR / file_name, config_type)

        assert isinstance(loaded.config, config_type)
        assert isinstance(loaded.metadata, dict)


def test_tiny_recipe_matches_test_fixture_and_keeps_run_metadata(tiny_stage1_config):
    loaded = load_config(CONFIG_DIR / "stage1_tiny_debug.json", Stage1Config)

    assert loaded.config == tiny_stage1_config
    assert loaded.metadata["max_steps"] == 2
    assert loaded.metadata["checkpoint_every"] == 1
    assert "config" not in loaded.metadata


def test_tiny_stage2_and_inference_recipes_match_test_fixtures(
    tiny_stage2_config,
    tiny_inference_config,
):
    stage2 = load_config(CONFIG_DIR / "stage2_tiny_debug.json", Stage2Config)
    inference = load_config(
        CONFIG_DIR / "inference_tiny_debug.json",
        InferenceConfig,
    )

    assert stage2.config == tiny_stage2_config
    assert inference.config == tiny_inference_config


def test_paper_recipes_match_default_configs():
    stage1 = load_config(CONFIG_DIR / "stage1_paper.json", Stage1Config)
    stage2 = load_config(CONFIG_DIR / "stage2_paper.json", Stage2Config)

    assert stage1.config == Stage1Config()
    assert stage2.config == Stage2Config()
    assert stage1.metadata["source_document"] == CONTEXT_DOC
    assert stage2.metadata["source_document"] == CONTEXT_DOC


def test_unknown_nested_config_keys_fail_clearly():
    with pytest.raises(ValueError, match="Unknown keys for vae: hidden_width"):
        config_from_dict(
            Stage1Config,
            {
                "vae": {
                    "hidden_width": 32,
                },
            },
        )


def test_unknown_top_level_keys_fail_without_recipe_wrapper():
    with pytest.raises(ValueError, match="Unknown keys for Stage1Config: max_steps"):
        config_from_dict(Stage1Config, {"max_steps": 2})


def test_invalid_config_values_fail_clearly():
    with pytest.raises(ValueError, match="global_batch_size must be positive"):
        config_from_dict(Stage1Config, {"global_batch_size": 0})


def test_invalid_config_value_types_fail_clearly():
    with pytest.raises(TypeError, match="optimizer.peak_lr must be a float"):
        config_from_dict(Stage1Config, {"optimizer": {"peak_lr": "fast"}})
