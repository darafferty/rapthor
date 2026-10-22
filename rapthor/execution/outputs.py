"""Shared output discovery helpers for execution tasks."""

import glob
import hashlib
import json
import logging
import os
import shutil
import stat
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

from rapthor.lib.records import directory_record, file_record

log = logging.getLogger("rapthor:outputs")


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


def partial_product_path(path: str) -> str:
    """Return the hidden sibling path at which a product is made before promotion."""
    directory, name = os.path.split(path)
    return os.path.join(directory, f".{name}.partial")


def product_record_path(path: str) -> str:
    """Return the hidden sibling file that records how a promoted product was made."""
    directory, name = os.path.split(path)
    return os.path.join(directory, f".{name}.product.json")


def make_command_product(
    output_path: str,
    build_command: Callable[[str], Sequence[str]],
    run_command: Callable[[list[str]], object],
    *,
    description: str,
    inputs: Iterable[str] = (),
    directory: bool = False,
) -> dict:
    """
    Make a product with an external command, or reuse the one an earlier attempt made.

    The command writes to a hidden partial path, which is renamed to
    ``output_path`` only after the command has succeeded, so a product at the
    final path is complete. A record next to it holds a digest of the command
    and of its inputs. A product is reused only when that digest matches, so
    that a product made with other settings, from other inputs, from an input
    that has been made again, or by an earlier Rapthor version is made again.

    Parameters
    ----------
    output_path : str
        Final path of the product. It is owned by the caller and may be removed.
    build_command : callable
        Returns the command that writes the product to the path it is given.
    run_command : callable
        Runs the command and raises when it fails.
    description : str
        Product description used in log and error messages.
    inputs : iterable of str
        Paths that the command reads. They are never removed, and a product that
        was made from a different version of one of them is not reused.
    directory : bool
        Whether the product is a directory (such as a Measurement Set) rather
        than a file.

    Returns
    -------
    dict
        The File or Directory record of ``output_path``.
    """
    inputs = [str(path) for path in inputs]
    require = require_directory if directory else require_file
    partial_path = partial_product_path(output_path)
    for path in (output_path, partial_path):
        _check_not_input(path, inputs)
    # Only this function writes the partial path, so anything there was left by
    # an attempt that failed or was stopped before promotion.
    _remove_path(partial_path)
    command = [str(token) for token in build_command(partial_path)]
    digest = _product_digest(command, inputs)
    if _recorded_product_digest(output_path, directory) == digest:
        log.info("Using the %s made by an earlier attempt: %s", description, output_path)
        return require(output_path, description)

    record_path = product_record_path(output_path)
    _remove_path(record_path)
    if os.path.lexists(output_path):
        log.info("Removing the %s left by an earlier attempt: %s", description, output_path)
        _remove_path(output_path)
    run_command(command)
    require(partial_path, description)
    # Write the record before promotion: until the rename, the final path does
    # not exist, so the record cannot describe an older product.
    _write_json_atomically(
        record_path,
        {"digest": digest, "command": command, "inputs": inputs},
    )
    os.replace(partial_path, output_path)
    return require(output_path, description)


def _check_not_input(path: str, inputs: list[str]) -> None:
    """Refuse to write a path that is, contains or lies inside one of the inputs."""
    for candidate in {Path(os.path.abspath(path)), Path(os.path.realpath(path))}:
        for input_path in inputs:
            for resolved in {Path(os.path.abspath(input_path)), Path(os.path.realpath(input_path))}:
                if candidate.is_relative_to(resolved) or resolved.is_relative_to(candidate):
                    raise ValueError(f"Output {path} would overwrite the input {input_path}")


def _remove_path(path: str) -> None:
    """Remove a file, link or directory tree, without following links."""
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def _product_digest(command: list[str], inputs: list[str]) -> str:
    content = {
        "command": command,
        "inputs": [[path, _input_identity(path)] for path in inputs],
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _input_identity(path: str) -> object:
    """
    Return what identifies the version of an input beyond its path.

    A product made by :func:`make_command_product` is identified by its own
    digest. A regular file is identified by its size and modification time.
    Other inputs, such as the input Measurement Sets, are not changed by Rapthor
    and are identified by their path alone.
    """
    for is_directory in (True, False):
        digest = _recorded_product_digest(path, is_directory)
        if digest is not None:
            return digest
    try:
        status = os.stat(path)
    except OSError:
        return None
    if stat.S_ISREG(status.st_mode):
        return [status.st_size, status.st_mtime_ns]
    return None


def _recorded_product_digest(path: str, directory: bool) -> Optional[str]:
    """Return the recorded digest of a promoted product, or ``None``."""
    exists = os.path.isdir(path) if directory else os.path.isfile(path)
    if not exists:
        return None
    try:
        with open(product_record_path(path)) as handle:
            digest = json.load(handle).get("digest")
    except (OSError, ValueError, AttributeError):
        return None
    return digest if isinstance(digest, str) else None


def _write_json_atomically(path: str, content: Mapping[str, object]) -> None:
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w") as handle:
        json.dump(content, handle, indent=4, sort_keys=True)
    os.replace(temporary_path, path)


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
        _remove_path(product_record_path(str(path)))
