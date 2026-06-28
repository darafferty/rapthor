"""Shared output discovery helpers for execution tasks."""

import glob
import os
import shutil
from typing import Optional

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


def cleanup_directory(path: str) -> None:
    """Remove a temporary directory if it exists."""
    if os.path.isdir(path):
        shutil.rmtree(path)
