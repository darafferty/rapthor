"""Shell command wrappers used by Prefect tasks."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

from rapthor.execution.commands import CommandInput, command_to_string, normalize_command
from rapthor.execution.config import ExecutionConfig


class MissingPrefectShellError(RuntimeError):
    """Raised when prefect-shell is required but not installed."""


@dataclass(frozen=True)
class ShellCommand:
    """A shell command plus execution metadata."""

    command: CommandInput
    environment: Mapping[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None
    name: Optional[str] = None

    @property
    def command_string(self) -> str:
        return command_to_string(self.command)


def _load_shell_operation_cls():
    try:
        from prefect_shell import ShellOperation
    except ImportError as err:
        raise MissingPrefectShellError("prefect-shell is required to execute shell tasks") from err
    return ShellOperation


def shell_operation_kwargs(
    shell_command: ShellCommand,
    execution_config: ExecutionConfig,
) -> dict:
    """Build keyword arguments for `prefect_shell.ShellOperation`."""
    kwargs = {
        "commands": [shell_command.command_string],
        "stream_output": execution_config.stream_output,
    }
    if shell_command.environment:
        kwargs["env"] = dict(shell_command.environment)
    if shell_command.working_directory is not None:
        kwargs["working_dir"] = shell_command.working_directory
    return kwargs


def command_log_path(working_directory: Optional[str]) -> Optional[Path]:
    """Return the backend-neutral command log path for an operation workdir."""
    if working_directory is None:
        return None
    workdir = Path(working_directory)
    if workdir.parent.name != "pipelines":
        return None
    return workdir.parent.parent / "logs" / "commands.jsonl"


def write_command_log_record(
    shell_command: ShellCommand,
    execution_config: ExecutionConfig,
) -> Optional[Path]:
    """Append a structured command record for integration/equivalence assertions."""
    if not execution_config.log_commands:
        return None

    log_path = command_log_path(shell_command.working_directory)
    if log_path is None:
        return None

    cwd = None if shell_command.working_directory is None else str(shell_command.working_directory)
    record = {
        "backend": "prefect",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": Path(cwd).name if cwd is not None else None,
        "name": shell_command.name,
        "cwd": cwd,
        "command": normalize_command(shell_command.command),
        "command_string": shell_command.command_string,
        "environment": dict(shell_command.environment),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return log_path


def run_shell_command(
    shell_command: ShellCommand,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
):
    """Execute one command using `prefect_shell.ShellOperation`."""
    write_command_log_record(shell_command, execution_config)
    operation_cls = shell_operation_cls or _load_shell_operation_cls()
    operation = operation_cls(**shell_operation_kwargs(shell_command, execution_config))
    return operation.run()


def run_shell_commands(
    commands: Sequence[ShellCommand],
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> list:
    """Execute commands sequentially and return their results."""
    return [
        run_shell_command(command, execution_config, shell_operation_cls=shell_operation_cls)
        for command in commands
    ]
