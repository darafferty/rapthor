"""Output record helpers for operation finalizers."""

from pathlib import Path
from typing import Any, Mapping

OUTPUT_CLASSES = ("File", "Directory")


class OutputRecordError(ValueError):
    """Raised when an output record does not match the finalizer contract."""


def _path_string(path: Any) -> str:
    if isinstance(path, Path):
        return path.as_posix()
    return str(path)


def file_record(path: Any) -> dict:
    """Create a finalizer-compatible file output record."""
    return {"class": "File", "path": _path_string(path)}


def directory_record(path: Any) -> dict:
    """Create a finalizer-compatible directory output record."""
    return {"class": "Directory", "path": _path_string(path)}


def is_output_record(value: Any) -> bool:
    """Return True if *value* is a finalizer-compatible file/directory record."""
    return (
        isinstance(value, Mapping)
        and value.get("class") in OUTPUT_CLASSES
        and isinstance(value.get("path"), str)
        and bool(value.get("path"))
    )


def validate_output_record(value: Any, allow_none: bool = False) -> Any:
    """Validate nested finalizer-compatible output records.

    Lists are walked recursively because several operation outputs are arrays or
    nested arrays of files. The validated value is returned unchanged to make the
    helper convenient in flow code.
    """
    if value is None and allow_none:
        return value
    if isinstance(value, list):
        for item in value:
            validate_output_record(item, allow_none=allow_none)
        return value
    if is_output_record(value):
        return value
    raise OutputRecordError(f"Invalid output record: {value!r}")
