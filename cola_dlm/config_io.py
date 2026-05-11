"""JSON config and recipe IO for Cola DLM dataclass configs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Generic, Literal, TypeVar, Union, get_args, get_origin


ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class LoadedConfig(Generic[ConfigT]):
    """A loaded model config plus top-level run metadata from a recipe file."""

    config: ConfigT
    metadata: dict[str, Any]


def config_to_dict(config: Any) -> dict[str, Any]:
    """Return a JSON-serializable dictionary for a dataclass config."""

    if not _is_dataclass_instance(config):
        raise TypeError("config must be a dataclass instance")
    return {
        field.name: _to_jsonable(getattr(config, field.name))
        for field in fields(config)
    }


def config_from_dict(config_type: type[ConfigT], values: Mapping[str, Any]) -> ConfigT:
    """Build a dataclass config from a mapping with strict key validation."""

    _require_dataclass_type(config_type)
    if not isinstance(values, Mapping):
        raise TypeError(f"{config_type.__name__} values must be a mapping")
    return _config_from_mapping(config_type, values, path="")


def load_config(path: str | Path, config_type: type[ConfigT]) -> LoadedConfig[ConfigT]:
    """Load a JSON recipe as a typed config plus top-level metadata."""

    _require_dataclass_type(config_type)
    recipe_path = Path(path)
    _require_json_path(recipe_path)
    try:
        raw = json.loads(recipe_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {recipe_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config file is not valid JSON: {recipe_path}") from exc

    if not isinstance(raw, Mapping):
        raise TypeError("Config file must contain a JSON object")

    if "config" in raw:
        config_values = raw["config"]
        metadata = {key: value for key, value in raw.items() if key != "config"}
    else:
        config_values = raw
        metadata = {}

    return LoadedConfig(
        config=config_from_dict(config_type, config_values),
        metadata=dict(metadata),
    )


def save_config(
    config: Any,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save a dataclass config as JSON, optionally with top-level metadata."""

    config_values = config_to_dict(config)
    output_path = Path(path)
    _require_json_path(output_path)

    if metadata is None:
        payload: dict[str, Any] = config_values
    else:
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if "config" in metadata:
            raise ValueError("metadata must not contain the reserved key 'config'")
        payload = {"config": config_values, **dict(metadata)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_to_jsonable(payload), indent=2) + "\n",
        encoding="utf-8",
    )


def _config_from_mapping(
    config_type: type[ConfigT],
    values: Mapping[str, Any],
    *,
    path: str,
) -> ConfigT:
    field_by_name = {field.name: field for field in fields(config_type)}
    unknown_keys = sorted(set(values) - set(field_by_name))
    if unknown_keys:
        location = path or config_type.__name__
        keys = ", ".join(unknown_keys)
        raise ValueError(f"Unknown keys for {location}: {keys}")

    kwargs = {}
    for name, field in field_by_name.items():
        if name in values:
            kwargs[name] = _convert_value(
                field.type,
                values[name],
                path=_join_path(path, name),
            )

    try:
        return config_type(**kwargs)
    except ValueError as exc:
        raise ValueError(f"Invalid {config_type.__name__}: {exc}") from exc
    except TypeError as exc:
        raise TypeError(f"Invalid {config_type.__name__}: {exc}") from exc


def _convert_value(expected_type: Any, value: Any, *, path: str) -> Any:
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if expected_type is Any:
        return value
    if value is None:
        if _allows_none(expected_type):
            return None
        raise TypeError(f"{path} must not be null")
    if _is_dataclass_type(expected_type):
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be an object for {expected_type.__name__}")
        return _config_from_mapping(expected_type, value, path=path)
    if origin in (Union, UnionType):
        return _convert_union(expected_type, value, path=path)
    if origin is Literal:
        if value not in args:
            allowed = ", ".join(repr(option) for option in args)
            raise ValueError(f"{path} must be one of: {allowed}")
        return value
    if origin is tuple:
        return _convert_tuple(args, value, path=path)
    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"{path} must be a list")
        item_type = args[0] if args else Any
        return [
            _convert_value(item_type, item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if expected_type is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{path} must be a bool")
        return value
    if expected_type is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{path} must be an int")
        return value
    if expected_type is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{path} must be a float")
        return float(value)
    if expected_type is str:
        if not isinstance(value, str):
            raise TypeError(f"{path} must be a string")
        return value
    return value


def _convert_union(expected_type: Any, value: Any, *, path: str) -> Any:
    errors = []
    for option in get_args(expected_type):
        if option is type(None):
            continue
        try:
            return _convert_value(option, value, path=path)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    joined_errors = "; ".join(errors)
    raise TypeError(f"{path} does not match any allowed type: {joined_errors}")


def _convert_tuple(args: tuple[Any, ...], value: Any, *, path: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{path} must be a list or tuple")
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(
            _convert_value(args[0], item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if args and len(value) != len(args):
        raise ValueError(f"{path} must contain exactly {len(args)} values")
    return tuple(
        _convert_value(item_type, item, path=f"{path}[{index}]")
        for index, (item_type, item) in enumerate(zip(args, value))
    )


def _to_jsonable(value: Any) -> Any:
    if _is_dataclass_instance(value):
        return config_to_dict(value)
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _is_dataclass_type(value: Any) -> bool:
    return isinstance(value, type) and is_dataclass(value)


def _is_dataclass_instance(value: Any) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


def _require_dataclass_type(config_type: Any) -> None:
    if not _is_dataclass_type(config_type):
        raise TypeError("config_type must be a dataclass type")


def _require_json_path(path: Path) -> None:
    if path.suffix != ".json":
        raise ValueError("Only .json config files are supported")


def _allows_none(expected_type: Any) -> bool:
    return type(None) in get_args(expected_type)


def _join_path(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


__all__ = (
    "LoadedConfig",
    "config_to_dict",
    "config_from_dict",
    "load_config",
    "save_config",
)
