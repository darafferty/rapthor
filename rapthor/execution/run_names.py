"""Helpers for readable Prefect flow and task run names."""

import os
import re
from typing import Mapping, Optional

_CYCLE_SUFFIX = re.compile(r"_(\d+)$")
_RUN_NAME_SAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_run_name_part(value: object) -> str:
    text = str(value).strip()
    text = _RUN_NAME_SAFE_CHARACTERS.sub("_", text)
    return text.strip("_")


def operation_basename(payload: Mapping[str, object]) -> str:
    """Return the operation directory name from a flow payload."""
    pipeline_working_dir = str(payload.get("pipeline_working_dir", ""))
    return os.path.basename(os.path.normpath(pipeline_working_dir))


def operation_cycle(payload: Mapping[str, object]) -> Optional[int]:
    """Return the self-calibration cycle encoded in the operation directory."""
    match = _CYCLE_SUFFIX.search(operation_basename(payload))
    if match is None:
        return None
    return int(match.group(1))


def operation_run_name(
    payload: Mapping[str, object],
    operation: str,
    *,
    mode: Optional[object] = None,
) -> str:
    """Build a compact Prefect run name for an operation flow."""
    cycle = operation_cycle(payload)
    operation = _safe_run_name_part(operation)
    mode_part = _safe_run_name_part(mode) if mode else ""
    if cycle is not None and mode_part:
        return f"{operation}_{mode_part}_{cycle}"
    if cycle is not None:
        return f"{operation}_{cycle}"
    if mode_part:
        return f"{operation}_{mode_part}"
    return _safe_run_name_part(operation_basename(payload)) or operation


def task_run_name(*parts: object) -> str:
    """Build a readable local Prefect task run name below an operation flow."""
    name_parts = [_safe_run_name_part(part) for part in parts if part not in (None, "")]
    return "_".join(part for part in name_parts if part)
