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


def test_recipe_files_load_without_optional_dependencies():
    recipes = [
        ("stage1_tiny_debug.json", Stage1Config),
        ("stage2_tiny_debug.json", Stage2Config),
        ("inference_tiny_debug.json", InferenceConfig),
        ("stage1_paper.json", Stage1Config),
        ("stage2_paper.json", Stage2Config),
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
