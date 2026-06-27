"""Serializable task-payload validation helpers."""

from typing import Any, Mapping, TypedDict


class ConcatenateEpochPayload(TypedDict):
    """Serializable inputs and expected output for one concatenate epoch."""

    input_filenames: list[str]
    output_filename: str
    output_path: str


class ConcatenatePayload(TypedDict):
    """Serializable payload submitted to the concatenate flow."""

    pipeline_working_dir: str
    data_colname: str
    epochs: list[ConcatenateEpochPayload]


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
