import importlib
import sys


def test_package_import_is_lightweight():
    sys.modules.pop("cola_dlm", None)
    sys.modules.pop("cola_dlm.config", None)

    package = importlib.import_module("cola_dlm")

    assert package.__version__ == "0.1.0"
    assert "cola_dlm.config" not in sys.modules


def test_config_boundary_imports():
    config = importlib.import_module("cola_dlm.config")

    assert config.__all__ == ()
