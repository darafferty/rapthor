"""Helpers for resolving path-like values in Rapthor parset files."""

from __future__ import annotations

import configparser
import json
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

INTERNALLY_DEFINED_STRATEGIES = ["image", "selfcal"]


def is_empty_path_value(value: str) -> bool:
    """Return True when a parset path value represents an intentionally empty path."""
    return value.strip() in {"", "None", "none", "null", "Null"}


def is_name_of_internal_strategy(value: str) -> bool:
    """Return True when a parset path value represents an internally defined strategy."""
    return value.strip() in INTERNALLY_DEFINED_STRATEGIES


def resolve_path_token(value: str, base_dir: Path) -> str:
    """Resolve one path token without altering empty values, internal strategy names, URLs, or absolute paths."""
    token = value.strip()
    if is_empty_path_value(token):
        return value
    if is_name_of_internal_strategy(token.lower()):
        return token.lower()
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
        if parset_file.suffix == ".json":
            try:
                parset_dict = json.load(handle)
            except json.decoder.JSONDecodeError as err:
                raise ValueError(f"JSON parset could not be parsed correctly.\n{err}")
            if not isinstance(parset_dict, dict):
                raise ValueError(
                    "JSON parset could not be parsed correctly.\nThe top-level entry must "
                    "be a dictionary."
                )
            # Convert all entires to str
            for section in parset_dict:
                if not isinstance(parset_dict[section], dict):
                    raise ValueError(
                        "JSON parset could not be parsed correctly.\nAll secondary-level "
                        "entries must be dictionaries, one per parset section."
                    )
                parset_dict[section] = {
                    # Handle the lists separately to ensure the string version do not have
                    # nested quotes (e.g. we want "[one, two]" instead of '["one", "two"]')
                    k: str(v) if not isinstance(v, list) else "[" + ", ".join(map(str, v)) + "]"
                    for k, v in parset_dict[section].items()
                }
            parser.read_dict(parset_dict)
        else:
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
