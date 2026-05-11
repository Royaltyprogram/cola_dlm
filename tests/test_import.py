import importlib
import sys


IMPLEMENTATION_MODULE_PREFIXES = (
    "cola_dlm.data",
    "cola_dlm.diffusion",
    "cola_dlm.models",
    "cola_dlm.stage1",
    "cola_dlm.train",
    "cola_dlm.training",
    "cola_dlm.trainers",
)


def _is_implementation_module(module_name: str) -> bool:
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in IMPLEMENTATION_MODULE_PREFIXES
    )


def test_package_import_is_lightweight():
    sys.modules.pop("cola_dlm", None)
    sys.modules.pop("cola_dlm.config", None)

    package = importlib.import_module("cola_dlm")

    assert package.__version__ == "0.1.0"
    assert "cola_dlm.config" not in sys.modules


def test_config_boundary_imports():
    config = importlib.import_module("cola_dlm.config")

    assert config.__all__ == (
        "VAEConfig",
        "DiTConfig",
        "DiffusionConfig",
        "OptimizerConfig",
        "Stage1Config",
        "Stage2Config",
        "InferenceConfig",
    )


def test_config_object_imports_do_not_import_training_modules():
    for module_name in list(sys.modules):
        if _is_implementation_module(module_name):
            sys.modules.pop(module_name)
    sys.modules.pop("cola_dlm.config", None)

    config = importlib.import_module("cola_dlm.config")

    for name in config.__all__:
        getattr(config, name)

    imported_modules = set(sys.modules)
    assert not any(
        _is_implementation_module(module_name)
        for module_name in imported_modules
    )


def test_dit_imports_do_not_import_training_modules():
    for module_name in list(sys.modules):
        if _is_implementation_module(module_name):
            sys.modules.pop(module_name)
    sys.modules.pop("cola_dlm.dit", None)

    dit = importlib.import_module("cola_dlm.dit")

    assert dit.__all__ == ("TimestepEmbedding", "BlockCausalTextDiT")
    imported_modules = set(sys.modules)
    assert not any(
        _is_implementation_module(module_name)
        for module_name in imported_modules
    )
