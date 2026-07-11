"""Helpers for readable Prefect flow and task run metadata."""

import re
from pathlib import Path
from typing import Iterable, Mapping, Optional

_CYCLE_SUFFIX = re.compile(r"_(\d+)$")
_RUN_NAME_SAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_run_name_part(value: object) -> str:
    text = str(value).strip()
    text = _RUN_NAME_SAFE_CHARACTERS.sub("_", text)
    return text.strip("_")


def operation_basename(payload: Mapping[str, object]) -> str:
    """Return the operation directory name from a flow payload."""
    pipeline_working_dir = str(payload.get("pipeline_working_dir", ""))
    return Path(pipeline_working_dir).name


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


def task_tags(*tags: object) -> list[str]:
    """Return stable, sanitized Prefect task tags."""
    sanitized = [_safe_run_name_part(tag).lower() for tag in tags if tag not in (None, "")]
    return sorted(dict.fromkeys(tag for tag in sanitized if tag))


def task_run_options(
    *name_parts: object,
    tags: Optional[Iterable[object]] = None,
) -> dict[str, object]:
    """Build keyword arguments for ``Prefect task.with_options``."""
    options: dict[str, object] = {"task_run_name": task_run_name(*name_parts)}
    task_tag_values = task_tags(*(tags or ()))
    if task_tag_values:
        options["tags"] = task_tag_values
    return options
