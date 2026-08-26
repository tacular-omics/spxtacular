"""Shared helpers for the versioned JSON transport formats."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any, Literal, overload

import numpy as np

SPECTRUM_SCHEMA = "spxtacular.spectrum"
CHROMATOGRAM_SCHEMA = "spxtacular.chromatogram"
JSON_SCHEMA_VERSION = 1


def get_json_schema(kind: Literal["spectrum", "chromatogram"]) -> dict[str, Any]:
    """Return a fresh copy of a packaged v1 JSON Schema document."""
    schema_files = {
        "spectrum": "spectrum-v1.schema.json",
        "chromatogram": "chromatogram-v1.schema.json",
    }
    try:
        filename = schema_files[kind]
    except KeyError as error:
        raise ValueError(f"Unknown JSON Schema kind {kind!r}. Expected 'spectrum' or 'chromatogram'") from error
    value = json.loads(files("spxtacular.schemas").joinpath(filename).read_text(encoding="utf-8"))
    return dict(require_mapping(value, f"packaged schema {filename}"))


def to_json_value(value: Any, path: str = "value") -> Any:
    """Return ``value`` using only strict JSON-native types.

    NumPy values are common in reader-produced metadata. Converting them here
    keeps every public ``to_dict`` result directly acceptable to a standards-
    compliant JSON encoder.
    """
    if value is None:
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path} must be finite for JSON serialization")
        return number
    if isinstance(value, np.ndarray):
        return [to_json_value(item, f"{path}[{index}]") for index, item in enumerate(value.tolist())]
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings, got {type(key).__name__}")
            result[key] = to_json_value(item, f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} contains unsupported type {type(value).__name__}")


def strict_json_dumps(payload: Mapping[str, Any], indent: int | None = None) -> str:
    """Encode compact standards-compliant JSON unless indentation is requested."""
    options: dict[str, Any] = {"allow_nan": False, "ensure_ascii": False, "indent": indent}
    if indent is None:
        options["separators"] = (",", ":")
    return json.dumps(payload, **options)


def strict_json_loads(value: str | bytes | bytearray) -> Any:
    """Decode JSON while rejecting the non-standard NaN and Infinity tokens."""
    if not isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"JSON input must be str, bytes, or bytearray, got {type(value).__name__}")

    def reject_constant(constant: str) -> None:
        raise ValueError(f"JSON input contains non-standard numeric constant {constant}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"JSON input contains duplicate object key {key!r}")
            result[key] = item
        return result

    try:
        return json.loads(value, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid JSON: {error}") from error


def require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    """Require a JSON object at ``path``."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a JSON object")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{path} keys must be strings, got {type(key).__name__}")
    return value


def require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    """Require exactly the keys defined by a versioned schema object."""
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"{path} is missing required field(s): {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{path} contains unknown field(s): {', '.join(unknown)}")


def require_array_or_none(value: Any, path: str) -> list[Any] | None:
    """Require a JSON array or null, returning a shallow list copy."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a JSON array or null")
    return value.copy()


@overload
def require_number(value: Any, path: str, *, nullable: Literal[False] = False) -> int | float: ...


@overload
def require_number(value: Any, path: str, *, nullable: Literal[True]) -> int | float | None: ...


def require_number(value: Any, path: str, *, nullable: bool = False) -> int | float | None:
    """Require a finite JSON number, optionally accepting null."""
    if value is None:
        if nullable:
            return None
        raise TypeError(f"{path} must be a number")
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        suffix = " or null" if nullable else ""
        raise TypeError(f"{path} must be a number{suffix}")
    number = to_json_value(value, path)
    return number


@overload
def require_integer(value: Any, path: str, *, nullable: Literal[False] = False) -> int: ...


@overload
def require_integer(value: Any, path: str, *, nullable: Literal[True]) -> int | None: ...


def require_integer(value: Any, path: str, *, nullable: bool = False) -> int | None:
    """Require a JSON integer, optionally accepting null."""
    if value is None:
        if nullable:
            return None
        raise TypeError(f"{path} must be an integer")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        suffix = " or null" if nullable else ""
        raise TypeError(f"{path} must be an integer{suffix}")
    return int(value)


@overload
def require_string(value: Any, path: str, *, nullable: Literal[False] = False) -> str: ...


@overload
def require_string(value: Any, path: str, *, nullable: Literal[True]) -> str | None: ...


def require_string(value: Any, path: str, *, nullable: bool = False) -> str | None:
    """Require a JSON string, optionally accepting null."""
    if value is None:
        if nullable:
            return None
        raise TypeError(f"{path} must be a string")
    if not isinstance(value, str):
        suffix = " or null" if nullable else ""
        raise TypeError(f"{path} must be a string{suffix}")
    return str(value)


@overload
def require_boolean(value: Any, path: str, *, nullable: Literal[False] = False) -> bool: ...


@overload
def require_boolean(value: Any, path: str, *, nullable: Literal[True]) -> bool | None: ...


def require_boolean(value: Any, path: str, *, nullable: bool = False) -> bool | None:
    """Require a JSON boolean, optionally accepting null."""
    if value is None:
        if nullable:
            return None
        raise TypeError(f"{path} must be a boolean")
    if not isinstance(value, (bool, np.bool_)):
        suffix = " or null" if nullable else ""
        raise TypeError(f"{path} must be a boolean{suffix}")
    return bool(value)


def require_number_array_or_none(value: Any, path: str) -> list[int | float] | None:
    """Require an array of finite JSON numbers or null."""
    array = require_array_or_none(value, path)
    if array is None:
        return None
    return [require_number(item, f"{path}[{index}]") for index, item in enumerate(array)]


def require_integer_array_or_none(value: Any, path: str) -> list[int] | None:
    """Require an array of JSON integers or null."""
    array = require_array_or_none(value, path)
    if array is None:
        return None
    return [require_integer(item, f"{path}[{index}]") for index, item in enumerate(array)]


def require_range_or_none(value: Any, path: str) -> list[int | float] | None:
    """Require a two-number JSON array or null."""
    array = require_number_array_or_none(value, path)
    if array is not None and len(array) != 2:
        raise ValueError(f"{path} must contain exactly two numbers")
    return array


def require_schema(
    payload: Mapping[str, Any],
    schema: str,
    kinds: set[str],
) -> str:
    """Validate the common transport envelope and return its kind."""
    require_exact_keys(payload, {"schema", "schema_version", "kind", "arrays", "metadata"}, "payload")
    if not isinstance(payload["schema"], str) or payload["schema"] != schema:
        raise ValueError(f"Expected schema {schema!r}, got {payload['schema']!r}")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != JSON_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {schema} schema version {payload['schema_version']!r}. "
            f"This release supports version {JSON_SCHEMA_VERSION}"
        )
    kind = payload["kind"]
    if not isinstance(kind, str) or kind not in kinds:
        expected = ", ".join(sorted(kinds))
        raise ValueError(f"payload.kind must be one of {expected}, got {kind!r}")
    return str(kind)
