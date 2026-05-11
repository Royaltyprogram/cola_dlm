import json
from pathlib import Path

from cola_dlm.config import Stage2Config
from cola_dlm.config_io import load_config
from cola_dlm.stage2 import _resolve_stage2_weights


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
ABLATION_DIR = CONFIG_DIR / "ablations"
PAPER_BASE_EXTENDS = "../paper/stage2_paper_base.json"


ABLATION_RECIPES = (
    "latent_dim_64.json",
    "latent_dim_128.json",
    "block_size_1.json",
    "block_size_64.json",
    "block_size_128.json",
    "fixed_vae_logsnr_1_0.json",
    "fixed_vae_logsnr_1_5.json",
    "fixed_vae_logsnr_2_0.json",
    "no_bert_loss.json",
)

EXPECTED_ABLATION_OVERRIDES = {
    "latent_dim_64.json": {
        "vae": {"latent_dim": 64},
        "dit": {"latent_dim": 64},
    },
    "latent_dim_128.json": {
        "vae": {"latent_dim": 128},
        "dit": {"latent_dim": 128},
    },
    "block_size_1.json": {"dit": {"block_size": 1}},
    "block_size_64.json": {"dit": {"block_size": 64}},
    "block_size_128.json": {"dit": {"block_size": 128}},
    "fixed_vae_logsnr_1_0.json": {"vae": {"fixed_logsnr": 1.0}},
    "fixed_vae_logsnr_1_5.json": {"vae": {"fixed_logsnr": 1.5}},
    "fixed_vae_logsnr_2_0.json": {"vae": {"fixed_logsnr": 2.0}},
    "no_bert_loss.json": {"mask_loss_weight": 0.0},
}


def test_all_ablation_recipes_load_as_stage2_configs():
    for file_name in ABLATION_RECIPES:
        loaded = load_config(ABLATION_DIR / file_name, Stage2Config)

        assert isinstance(loaded.config, Stage2Config)
        assert loaded.metadata["source_document"].endswith("00_context.md")
        assert loaded.metadata["ablation_source_document"].endswith(
            "project_page.html"
        )


def test_all_ablation_recipes_extend_stage2_paper_base():
    for file_name in ABLATION_RECIPES:
        raw = json.loads((ABLATION_DIR / file_name).read_text(encoding="utf-8"))

        assert raw["extends"] == PAPER_BASE_EXTENDS


def test_all_ablation_recipes_contain_only_narrow_overrides():
    for file_name in ABLATION_RECIPES:
        raw = json.loads((ABLATION_DIR / file_name).read_text(encoding="utf-8"))

        assert raw["config"] == EXPECTED_ABLATION_OVERRIDES[file_name]


def test_latent_dimension_ablation_recipes_keep_vae_and_dit_equal():
    expected_dims = {
        "latent_dim_64.json": 64,
        "latent_dim_128.json": 128,
    }

    for file_name, expected_dim in expected_dims.items():
        config = load_config(ABLATION_DIR / file_name, Stage2Config).config

        assert config.vae.latent_dim == expected_dim
        assert config.dit.latent_dim == expected_dim


def test_block_size_ablation_recipes_divide_sequence_length():
    expected_block_sizes = {
        "block_size_1.json": 1,
        "block_size_64.json": 64,
        "block_size_128.json": 128,
    }

    for file_name, expected_block_size in expected_block_sizes.items():
        config = load_config(ABLATION_DIR / file_name, Stage2Config).config

        assert config.dit.block_size == expected_block_size
        assert config.dit.sequence_length % config.dit.block_size == 0


def test_fixed_logsnr_ablation_recipes_are_non_negative_and_base_is_learnable():
    base_config = load_config(CONFIG_DIR / "stage2_paper.json", Stage2Config).config
    expected_logsnrs = {
        "fixed_vae_logsnr_1_0.json": 1.0,
        "fixed_vae_logsnr_1_5.json": 1.5,
        "fixed_vae_logsnr_2_0.json": 2.0,
    }

    assert base_config.vae.fixed_logsnr is None
    for file_name, expected_logsnr in expected_logsnrs.items():
        config = load_config(ABLATION_DIR / file_name, Stage2Config).config

        assert isinstance(config.vae.fixed_logsnr, float)
        assert config.vae.fixed_logsnr >= 0.0
        assert config.vae.fixed_logsnr == expected_logsnr


def test_no_bert_loss_ablation_resolves_stage2_mask_weight_to_zero():
    config = load_config(ABLATION_DIR / "no_bert_loss.json", Stage2Config).config

    *_, lambda_mask = _resolve_stage2_weights(
        stage2_config=config,
        lambda_vae=None,
        lambda_flow_matching=None,
        lambda_reference_kl=None,
        lambda_posterior_regularizer=None,
        lambda_mask=None,
    )

    assert config.mask_loss_weight == 0.0
    assert lambda_mask == 0.0
