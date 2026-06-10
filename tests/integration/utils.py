import configparser
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rapthor.execution.commands import normalize_command

COMMAND_LOG_FILENAME = "commands.jsonl"


@dataclass(frozen=True)
class CommandRecord:
    """A backend-neutral external command record from a Rapthor integration run."""

    operation: Optional[str]
    command: list[str]
    command_string: str
    source: Path
    backend: str = "unknown"
    name: Optional[str] = None

    @property
    def executable(self):
        return Path(self.command[0]).name if self.command else ""

    @property
    def arguments(self):
        return parse_command_arguments(self.command)


def get_working_dir_from_parset(parset_path):
    """Return dir_working from a parset file."""
    parset = configparser.ConfigParser()
    parset.read(parset_path)
    return parset["global"]["dir_working"]


def _operation_from_log_path(log_root, path):
    try:
        relative = path.relative_to(log_root)
    except ValueError:
        return None
    return relative.parts[0] if len(relative.parts) > 1 else None


def _command_records_from_jsonl(log_path):
    records = []
    if not log_path.exists():
        return records
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        command = normalize_command(item["command"])
        records.append(
            CommandRecord(
                operation=item.get("operation"),
                command=command,
                command_string=item.get("command_string") or shlex.join(command),
                source=log_path,
                backend=item.get("backend", "prefect"),
                name=item.get("name"),
            )
        )
    return records


def _extract_shell_commands(text):
    pattern = re.compile(r"^\$ (?P<command>.+?)(?=^\S|\Z)", re.MULTILINE | re.DOTALL)
    commands = []
    for match in pattern.finditer(text):
        command = match.group("command").replace("\\\n", " ").strip()
        if command:
            commands.append(command)
    return commands


def _command_records_from_legacy_logs(log_root):
    records = []
    if not log_root.exists():
        return records
    for log_path in sorted(log_root.glob("**/*.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for command_string in _extract_shell_commands(text):
            command = shlex.split(command_string)
            records.append(
                CommandRecord(
                    operation=_operation_from_log_path(log_root, log_path),
                    command=command,
                    command_string=shlex.join(command),
                    source=log_path,
                    backend="legacy-log",
                )
            )
    return records


def collect_command_records(working_dir):
    """Return external command records from Prefect JSONL and retained legacy logs."""
    working_dir = Path(working_dir)
    log_root = working_dir / "logs"
    records = _command_records_from_jsonl(log_root / COMMAND_LOG_FILENAME)
    records.extend(_command_records_from_legacy_logs(log_root))
    return records


def find_command_records(
    working_dir,
    *,
    operation=None,
    executable=None,
    contains=None,
):
    """Filter backend-neutral command records for integration assertions."""
    records = collect_command_records(working_dir)
    if operation is not None:
        records = [record for record in records if record.operation == operation]
    if executable is not None:
        records = [record for record in records if record.executable == executable]
    if contains is not None:
        records = [
            record
            for record in records
            if contains in record.command_string or contains in record.command
        ]
    return records


def first_command_arguments(
    working_dir,
    *,
    operation,
    executable,
    contains=None,
):
    """Return parsed arguments from the first matching command record."""
    records = find_command_records(
        working_dir,
        operation=operation,
        executable=executable,
        contains=contains,
    )
    assert records, f"Expected command {executable!r} for operation {operation!r}" + (
        "" if contains is None else f" containing {contains!r}"
    )
    return records[0].arguments


def update_parset_path(parset_path, param_dict):
    """Helper function to update parset parameters and return a new path."""
    parset = configparser.ConfigParser()
    parset.read(parset_path)
    missing_params = set(param_dict.keys())

    for section in parset.sections():
        for key, value in param_dict.items():
            if key in parset[section]:
                parset[section][key] = value
                missing_params.discard(key)

    updated_parset_path = parset_path.parent / "updated.parset"

    if missing_params:
        raise ValueError(f"Parameters {missing_params} not found in parset.")

    with updated_parset_path.open("w") as fp:
        parset.write(fp)
    return updated_parset_path


def get_wsclean_output_mtimes(image_pipeline_dir):
    """Return a mapping of WSClean output product filenames to their modification timestamps"""
    products = {}
    for pattern in [
        "*-MFS-image.fits",
        "*-MFS-image-pb.fits",
        "*-MFS-residual.fits",
        "*-MFS-dirty.fits",
    ]:
        for path in Path(image_pipeline_dir).glob(pattern):
            products[path.name] = path.stat().st_mtime_ns
    return products


def make_failing_filter_skymodel(fake_bin_dir):
    """Create a PATH-injected wrapper for filter_skymodel.py."""
    fake_script = fake_bin_dir / "filter_skymodel.py"
    fake_script.write_text("#!/usr/bin/env python3\nraise SystemExit(1)")
    fake_script.chmod(0o755)
    return fake_script


def parse_command_arguments(command):
    """Parse key-value arguments from a command token list or shell string."""
    tokens = normalize_command(command)
    return {
        key: value for key, _, value in (token.partition("=") for token in tokens if "=" in token)
    }
