"""Shared output discovery helpers for execution tasks."""

import glob
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

from rapthor._version import __version__
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
    ``output_path`` once the command has succeeded, so a product at the final
    path is complete. A record next to it holds a digest of the Rapthor version,
    the program, the command and the inputs, and the product is reused only when
    that digest matches.

    Parameters
    ----------
    output_path : str
        Final path of the product, which may be removed and made again.
    build_command : callable
        Returns the command that writes the product to the path it is given.
    run_command : callable
        Runs a command and raises when it fails.
    description : str
        Description of the product for log and error messages.
    inputs : iterable of str
        Paths that the command reads. They are never removed, and a product made
        from another version of one of them is not reused.
    directory : bool
        Whether the product is a directory (such as a Measurement Set).

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
    # an attempt that failed or was stopped before promotion
    _remove_path(partial_path)
    command = [str(token) for token in build_command(partial_path)]
    digest = _product_digest(command, inputs)
    exists = os.path.isdir(output_path) if directory else os.path.isfile(output_path)
    if exists and _recorded_product_digest(output_path) == digest:
        log.info("Using the %s made by an earlier attempt: %s", description, output_path)
        return require(output_path, description)

    record_path = product_record_path(output_path)
    _remove_path(record_path)
    if os.path.lexists(output_path):
        log.info("Removing the %s left by an earlier attempt: %s", description, output_path)
        _remove_path(output_path)
    run_command(command)
    require(partial_path, description)
    # Until the rename, the final path does not exist, so the record cannot
    # describe an older product
    _write_json_atomically(
        record_path,
        {
            "digest": digest,
            "command": command,
            "inputs": inputs,
            "files": [entry[0] for entry in _file_entries(partial_path)],
        },
    )
    os.replace(partial_path, output_path)
    return require(output_path, description)


def _check_not_input(path: str, inputs: list[str]) -> None:
    """Refuse to write a path that is, contains or lies inside one of the inputs."""
    for input_path in inputs:
        if any(
            output.is_relative_to(source) or source.is_relative_to(output)
            for output in _aliases(path)
            for source in _aliases(input_path)
        ):
            raise ValueError(f"Output {path} would overwrite the input {input_path}")


def _aliases(path: str) -> set[Path]:
    """Return the absolute path and the path with links resolved."""
    return {Path(os.path.abspath(path)), Path(os.path.realpath(path))}


def _remove_path(path: str) -> None:
    """Remove a file, link or directory tree, without following links."""
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def _product_digest(command: list[str], inputs: list[str]) -> str:
    """
    Return a digest of what makes a product: Rapthor, the program, the command and the inputs.

    An update of Rapthor or of the program (such as DP3 or WSClean) may change
    what the same command makes, so a product made before it is not reused.
    """
    content = {
        "rapthor": __version__,
        "program": _program_identity(command[0]),
        "command": command,
        "inputs": [[path, _input_identity(path)] for path in inputs],
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _program_identity(program: str) -> object:
    """Return the path, size and modification time of the file that runs ``program``."""
    path = shutil.which(program)
    return [path, *_file_entries(path)] if path else None


def _input_identity(path: str) -> object:
    """
    Return what identifies the version of an input beyond its path.

    A product of :func:`make_command_product` is identified by its digest, any
    other file or directory (such as an input Measurement Set) by the relative
    path, size and modification time of each of its files.
    """
    return _recorded_product_digest(path) or _file_entries(path)


def _file_entries(path: str) -> list[list]:
    """Return the relative path, size and modification time of each file in ``path``."""
    if os.path.isfile(path):
        status = os.stat(path)
        return [["", status.st_size, status.st_mtime_ns]]
    entries = []
    visited = set()
    for root, directories, names in os.walk(path, followlinks=True):
        # Follow linked subdirectories (such as linked subtables), but only once
        real_root = os.path.realpath(root)
        if real_root in visited:
            directories.clear()
            continue
        visited.add(real_root)
        directories.sort()
        for name in sorted(names):
            # Casacore may update the lock file of a table that it only reads
            if name == "table.lock":
                continue
            file_path = os.path.join(root, name)
            try:
                status = os.stat(file_path)
            except OSError:
                continue
            entries.append([os.path.relpath(file_path, path), status.st_size, status.st_mtime_ns])
    return entries


def _recorded_product_digest(path: str) -> Optional[str]:
    """
    Return the recorded digest of a promoted product that is still complete, or None.

    A directory product is complete if every file that it held when it was
    promoted still exists, so that one that was partly removed (for example by
    a cleanup or a move that was stopped) is not reused. Files added later, such
    as a column that WSClean adds to a Measurement Set, are allowed.
    """
    try:
        with open(product_record_path(path)) as handle:
            record = json.load(handle)
        digest, files = record["digest"], record["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(digest, str) or not isinstance(files, list):
        return None
    if os.path.isdir(path):
        if not all(os.path.isfile(os.path.join(path, str(name))) for name in files):
            return None
    elif not os.path.isfile(path):
        return None
    return digest


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
        # Remove the record first, so that a partly removed product is not reused
        _remove_path(product_record_path(str(path)))
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
