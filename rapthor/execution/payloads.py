"""Shared serializable task-payload validation helpers."""

import os
from typing import Any, Mapping, Optional


class PayloadSerializationError(TypeError):
    """Raised when a task payload is not safe to send to a Dask worker."""


def assert_serializable_payload(value: Any, path: str = "payload") -> Any:
    """Validate that *value* contains only simple serializable payload values.

    Task payloads intentionally use strings for paths and plain Python
    containers. This avoids passing live Rapthor domain objects to Dask workers.
    The validated value is returned unchanged for convenient inline use.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_serializable_payload(item, f"{path}[{index}]")
        return value

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PayloadSerializationError(f"{path} has non-string key {key!r}")
            assert_serializable_payload(item, f"{path}.{key}")
        return value

    raise PayloadSerializationError(
        f"{path} contains unsupported value {value!r} of type {type(value).__name__}"
    )


def validate_basename(filename: object, name: str) -> str:
    """Return a non-empty basename or raise a stable payload validation error."""
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{name} must be a non-empty string")
    if os.path.isabs(filename) or os.path.basename(filename) != filename:
        raise ValueError(f"{name} must be a basename")
    return filename


def validate_string_list(
    values: object,
    name: str,
    *,
    allow_empty: bool = True,
) -> list[str]:
    """Return a list of non-empty strings or raise a stable validation error."""
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        if not allow_empty:
            raise ValueError(f"{name} must be a non-empty list of strings")
        raise ValueError(f"{name} must be a list of strings")
    if not allow_empty and not values:
        raise ValueError(f"{name} must be a non-empty list of strings")
    return list(values)


def validate_int_list(values: object, name: str, length: Optional[int] = None) -> list[int]:
    """Return a list of integers, optionally requiring an exact length."""
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise ValueError(f"{name} must be a list of integers")
    if length is not None and len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} entries")
    return list(values)
