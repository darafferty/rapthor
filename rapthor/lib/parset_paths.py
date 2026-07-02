"""Helpers for resolving path-like values in Rapthor parset files."""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Optional

PARSET_PATH_OPTIONS = {
    "global": {
        "dir_working",
        "input_ms",
        "input_skymodel",
        "apparent_skymodel",
        "strategy",
        "input_h5parm",
        "input_fulljones_h5parm",
        "input_normalization_h5parm",
        "facet_layout",
    },
    "imaging": {
        "photometry_skymodel",
        "astrometry_skymodel",
        "normalization_skymodels",
    },
}


def is_empty_path_value(value: str) -> bool:
    """Return True when a parset path value represents an intentionally empty path."""
    return value.strip() in {"", "None", "none", "null", "Null"}


def resolve_path_token(value: str, base_dir: Path) -> str:
    """Resolve one path token without altering empty values, URLs, or absolute paths."""
    token = value.strip()
    if is_empty_path_value(token):
        return value
    if "://" in token:
        return token
    path = Path(token).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve(strict=False))


def resolve_path_value(value: str, base_dir: Path) -> str:
    """Resolve a scalar path value or a simple list of path values."""
    stripped = value.strip()
    if is_empty_path_value(stripped):
        return value
    if stripped.startswith("[") and stripped.endswith("]"):
        resolved = [
            resolve_path_token(token, base_dir)
            for token in stripped.strip("[]").split(",")
            if token.strip()
        ]
        return f"[{', '.join(resolved)}]"
    return resolve_path_token(value, base_dir)


def materialize_parset_paths(
    parset_file: Path,
    output_file: Path,
    *,
    working_dir_override: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """Write a parset copy with known path-like options resolved to absolute paths."""
    parser = configparser.ConfigParser(interpolation=None)
    with parset_file.open() as handle:
        parser.read_file(handle)

    path_base = Path.cwd() if base_dir is None else base_dir
    for section, options in PARSET_PATH_OPTIONS.items():
        if not parser.has_section(section):
            continue
        for option in options:
            if parser.has_option(section, option):
                parser.set(
                    section, option, resolve_path_value(parser.get(section, option), path_base)
                )

    if working_dir_override is not None:
        if not parser.has_section("global"):
            parser.add_section("global")
        parser.set("global", "dir_working", str(working_dir_override))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as handle:
        parser.write(handle)
    return output_file
