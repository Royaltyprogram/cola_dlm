import ast
import json
import math
from pathlib import Path

import pytest

from cola_dlm.modal_gpu_smoke import (
    build_tiny_modal_smoke_config,
    run_tiny_modal_gpu_validation,
    run_tiny_stage2_smoke_step,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODAL_SCRIPT = REPO_ROOT / "scripts" / "modal_gpu_smoke.py"


def test_build_tiny_modal_smoke_config_uses_expected_small_dimensions():
    config = build_tiny_modal_smoke_config()

    assert config.vae.sequence_length == 8
    assert config.dit.sequence_length == 8
    assert config.dit.block_size == 2
    assert config.vae.vocab_size == 32
    assert config.vae.latent_dim == 2
    assert config.dit.latent_dim == 2
    assert config.vae.encoder_layers == 1
    assert config.vae.decoder_layers == 1
    assert config.dit.num_layers == 1


def test_tiny_stage2_smoke_step_runs_one_cpu_step():
    result = run_tiny_stage2_smoke_step()

    assert result["success"] is True
    assert result["device"] == "cpu"
    assert isinstance(result["cuda_available"], bool)
    assert isinstance(result["loss"], float)
    assert math.isfinite(result["loss"])
    assert result["steps"] == 1


def test_tiny_stage2_smoke_step_reports_cpu_placement():
    result = run_tiny_stage2_smoke_step()

    assert result["vae_parameter_device"] == "cpu"
    assert result["dit_parameter_device"] == "cpu"
    assert result["token_device"] == "cpu"


def test_tiny_stage2_smoke_step_requires_cuda_when_requested():
    with pytest.raises(RuntimeError, match="CUDA is required"):
        run_tiny_stage2_smoke_step(device="cpu", require_cuda=True)


def test_tiny_modal_gpu_validation_runs_broader_cpu_checks():
    result = run_tiny_modal_gpu_validation()

    assert result["success"] is True
    assert result["device"] == "cpu"
    assert isinstance(result["cuda_available"], bool)
    assert result["limitations"] == []
    assert set(result["checks"]) == {
        "text_vae",
        "stage2_joint_training",
        "inference_generate",
    }

    text_vae = result["checks"]["text_vae"]
    assert text_vae["success"] is True
    assert text_vae["loss_finite"] is True
    assert math.isfinite(text_vae["loss"])
    assert text_vae["devices_match"] is True
    assert text_vae["trainable_parameter_changed"] is True

    stage2 = result["checks"]["stage2_joint_training"]
    assert stage2["success"] is True
    assert stage2["loss_finite"] is True
    assert math.isfinite(stage2["total_loss"])
    assert stage2["devices_match"] is True
    assert stage2["trainable_parameter_changed"] is True
    assert stage2["steps"] == 1

    inference = result["checks"]["inference_generate"]
    assert inference["success"] is True
    assert inference["skipped"] is False
    assert inference["devices_match"] is True


def test_tiny_modal_gpu_validation_reports_cpu_placement():
    result = run_tiny_modal_gpu_validation()
    text_vae = result["checks"]["text_vae"]
    stage2 = result["checks"]["stage2_joint_training"]
    inference = result["checks"]["inference_generate"]

    for name in (
        "loss_tensor_device",
        "vae_parameter_device",
        "token_device",
        "mask_device",
        "posterior_device",
        "posterior_mu_device",
        "posterior_logvar_device",
        "latent_device",
        "logits_device",
    ):
        assert text_vae[name] == "cpu"

    for name in (
        "loss_tensor_device",
        "vae_parameter_device",
        "reference_encoder_parameter_device",
        "dit_parameter_device",
        "token_device",
        "mask_device",
    ):
        assert stage2[name] == "cpu"

    for name in (
        "generated_latent_device",
        "all_latent_device",
        "response_logits_device",
        "response_token_device",
        "token_device",
        "mask_device",
        "vae_parameter_device",
        "dit_parameter_device",
    ):
        assert inference[name] == "cpu"


def test_tiny_modal_gpu_validation_requires_cuda_when_requested():
    with pytest.raises(RuntimeError, match="CUDA is required"):
        run_tiny_modal_gpu_validation(device="cpu", require_cuda=True)


def test_tiny_modal_gpu_validation_result_is_json_serializable():
    result = run_tiny_modal_gpu_validation()

    encoded = json.dumps(result, sort_keys=True)

    assert '"success": true' in encoded


def test_modal_gpu_smoke_script_declares_modal_entrypoint_without_importing_modal():
    source = MODAL_SCRIPT.read_text()
    tree = ast.parse(source)

    function_decorators = _decorators_named(tree, "function")
    local_entrypoint_decorators = _decorators_named(tree, "local_entrypoint")

    assert len(function_decorators) == 1
    assert len(local_entrypoint_decorators) == 1
    assert _calls_modal_app(tree)
    assert '.pip_install("torch")' in source
    assert '.add_local_python_source("cola_dlm")' in source

    function_kwargs = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in function_decorators[0].keywords
        if keyword.arg != "image"
    }
    assert function_kwargs["gpu"] == "T4"
    assert function_kwargs["max_containers"] == 1
    assert function_kwargs["min_containers"] == 0
    assert function_kwargs["scaledown_window"] == 30
    assert function_kwargs["timeout"] == 300


def _decorators_named(tree: ast.AST, name: str) -> list[ast.Call]:
    decorators: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == name
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "app"
            ):
                decorators.append(decorator)
    return decorators


def _calls_modal_app(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "App"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "modal"
        for node in ast.walk(tree)
    )
