"""Helpers for Rapthor file and directory output records."""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

logger = logging.getLogger("rapthor:records")

OUTPUT_CLASSES = ("File", "Directory")


class OutputRecordError(ValueError):
    """Raised when an output record does not match the finalizer contract."""


class PathRecord(object):
    """
    File or directory path record.

    Parameters
    ----------
    path : str or list of str
        Path or list of paths
    path_type : str
        Type of path: 'file' or 'directory'.
    """

    def __init__(self, path, path_type):
        if not isinstance(path, (str, list)):
            raise ValueError(
                f"path must be a string or a list of strings: obtained {type(path)} for {path_type}"
            )
        self.path = path
        if path_type.lower() == "file":
            self.path_type = "File"
        elif path_type.lower() == "directory":
            self.path_type = "Directory"
        else:
            raise ValueError('path type must be one of "file" or "directory"')

    def to_json(self):
        """
        Returns a dict suitable for use with json.dumps()
        """
        if type(self.path) is str:
            record_value = {"class": self.path_type, "path": self.path}
        else:
            record_value = []
            for p in self.path:
                record_value.append({"class": self.path_type, "path": p})

        return record_value


class FileRecord(PathRecord):
    """
    File record class.

    Parameters
    ----------
    filename : str or list of str
        Filename or list of filenames
    """

    def __init__(self, filename):
        super(FileRecord, self).__init__(filename, "file")


class DirectoryRecord(PathRecord):
    """
    Directory record class.

    Parameters
    ----------
    dirname : str or list of str
        Directory name or list of directory names
    """

    def __init__(self, dirname):
        super(DirectoryRecord, self).__init__(dirname, "directory")


class NpEncoder(json.JSONEncoder):
    """
    Numpy to JSON encoder class

    Numpy types cannot be serialized to JSON by default, so this
    class is used in json.dumps() calls when numpy types are
    present
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def is_file_record(record):
    """
    Check if the given object is a file record.

    A file record is a dictionary with the 'class' key set to 'File'.
    """
    return isinstance(record, dict) and record.get("class") == "File"


def is_directory_record(record):
    """
    Check if the given object is a directory record.

    A directory record is a dictionary with the 'class' key set to 'Directory'.
    """
    return isinstance(record, dict) and record.get("class") == "Directory"


def is_file_or_directory_record(record):
    """
    Check if the given object is a file or directory record.
    """
    return is_file_record(record) or is_directory_record(record)


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
    helper convenient in flow and finalizer code.
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


def output_record_path(record: Any, record_class: str) -> str:
    """Return the path from a required finalizer-compatible output record."""
    if isinstance(record, Mapping) and record.get("class") == record_class:
        path = record.get("path")
        if isinstance(path, str) and path:
            return path
    raise OutputRecordError(f"Expected a {record_class} output record, got {record!r}")


def optional_output_record_path(record: Any, record_class: str) -> Optional[str]:
    """Return the path from an optional output record, or ``None``."""
    if record is None:
        return None
    return output_record_path(record, record_class)


def file_record_path(record: Any) -> str:
    """Return the path from a required file output record."""
    return output_record_path(record, "File")


def optional_file_record_path(record: Any) -> Optional[str]:
    """Return the path from an optional file output record, or ``None``."""
    return optional_output_record_path(record, "File")


def directory_record_path(record: Any) -> str:
    """Return the path from a required directory output record."""
    return output_record_path(record, "Directory")


def optional_directory_record_path(record: Any) -> Optional[str]:
    """Return the path from an optional directory output record, or ``None``."""
    return optional_output_record_path(record, "Directory")


def copy_record_object(src_obj, dest_dir, move=False):
    """
    Copy a file or directory record to the specified destination directory.

    Parameters
    ----------
    src_obj : object
        Source object of the copy
    dest_dir: str
        Path of destination directory to which src_obj will be copied
    move : bool, optional
        If True, move files instead of copying them
    """
    if is_file_or_directory_record(src_obj) and os.path.exists(src_obj["path"]):
        os.makedirs(dest_dir, exist_ok=True)
        src = Path(src_obj["path"])
        dest = Path(dest_dir) / src.name
        if move:
            shutil.move(src, dest)
        else:
            if is_file_record(src_obj):
                shutil.copy(src, dest)
            elif is_directory_record(src_obj):
                shutil.copytree(src, dest, dirs_exist_ok=True)
    # Otherwise, do nothing


def copy_record_recursive(src_obj, dest_dir, index=None, move=False):
    """
    Recursively copy file or directory records to the specified destination
    directory.

    Parameters
    ----------
    src_obj : object or list of objects
        Source object(s) of the copy
    dest_dir: str
        Path of destination directory to which src_obj will be copied
    index : int
        If src_obj is a list and index is specified, only the item with the specified index is
        copied (other items in the list are ignored)
    move : bool, optional
        If True, move files instead of copying them
    """
    if isinstance(src_obj, list):
        for i, item in enumerate(src_obj):
            if index is None or i == index:
                copy_record_recursive(item, dest_dir, None, move)
    elif is_file_or_directory_record(src_obj):
        copy_record_object(src_obj, dest_dir, move)
    # Otherwise, do nothing


def remove_or_log_error(path: Path):
    """
    Remove a file or directory at the specified path.
    Log a warning if the file or directory does not exist.

    Parameters
    ----------
    path: Path object
        Path of file or directory to remove
    """
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except FileNotFoundError:
        logger.warning("Cannot remove non-existing path: %s", path)


def clean_if_file_or_directory_record(src_obj):
    """
    Remove file or directory records from the filesystem.

    Parameters
    ----------
    src_obj : object or list of objects
        Source object(s) to be removed
    """
    if isinstance(src_obj, list):
        for item in src_obj:
            clean_if_file_or_directory_record(item)
    elif is_file_or_directory_record(src_obj):
        remove_or_log_error(Path(src_obj["path"]))
    # Otherwise, do nothing
