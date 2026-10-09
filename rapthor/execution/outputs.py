"""Shared output discovery helpers for execution tasks."""

import glob
import os
import shutil
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

from rapthor.lib.records import directory_record, file_record


def require_file(path: str, description: str) -> dict:
    """Return a File record for a required output path."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} was not created: {path}")
    return file_record(path)


def require_directory(path: str, description: str) -> dict:
    """Return a Directory record for a required output path."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{description} was not created: {path}")
    return directory_record(path)


def output_path(output_dir: Optional[str], filename: str) -> str:
    """Return an output path, keeping cwd-relative script behavior when no directory is supplied."""
    if output_dir is None:
        return filename
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def first_existing_file(patterns: list[str], description: str) -> dict:
    """Return the first existing file record matching one of the patterns."""
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path):
                return file_record(path)
    raise FileNotFoundError(f"{description} was not created: {', '.join(patterns)}")


def optional_first_existing_file(patterns: list[str]) -> Optional[dict]:
    """Return the first matching file record, or ``None`` when nothing exists."""
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path):
                return file_record(path)
    return None


def file_records_for_patterns(patterns: list[str]) -> list[dict]:
    """Return all file records matching the supplied glob patterns."""
    records = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path):
                records.append(file_record(path))
    return records


def file_records_for_required_patterns(patterns: list[str], description: str) -> list[dict]:
    """Return all matching file records and fail when no file was produced."""
    records = file_records_for_patterns(patterns)
    if not records:
        raise FileNotFoundError(f"{description} was not created: {', '.join(patterns)}")
    return records


def compressed_file_record(record: dict, description: str) -> dict:
    """Return a File record for the compressed version of an output record."""
    return require_file(f"{record['path']}.fz", description)


def cleanup_intermediate_outputs(
    paths: Iterable[Union[str, Path]],
    working_directory: str,
    retained_outputs: object,
    *,
    input_paths: Iterable[Union[str, Path]] = (),
) -> None:
    """Remove owned intermediates while protecting inputs, outputs and their targets."""
    retained_paths = list(input_paths)

    def retain_records(value):
        if isinstance(value, Mapping):
            if value.get("class") in {"File", "Directory"}:
                retained_paths.append(value["path"])
            for item in value.values():
                retain_records(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                retain_records(item)

    retain_records(retained_outputs)
    retained = set()
    for item in retained_paths:
        path = Path(os.path.abspath(item))
        retained.update((path, path.resolve()))
    root = Path(os.path.abspath(working_directory))
    resolved_root = root.resolve()
    for candidate in paths:
        path = Path(os.path.abspath(candidate))
        if path == root or not path.is_relative_to(root):
            continue
        if not path.parent.resolve().is_relative_to(resolved_root):
            continue
        aliases = (path, path.resolve())
        if any(
            alias.is_relative_to(output) or output.is_relative_to(alias)
            for alias in aliases
            for output in retained
        ):
            continue
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
