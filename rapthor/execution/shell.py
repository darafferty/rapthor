"""Shell command wrappers used by Prefect tasks."""

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from rapthor.execution.commands import CommandInput, command_to_string
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
        kwargs["cwd"] = shell_command.working_directory
    return kwargs


def run_shell_command(
    shell_command: ShellCommand,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
):
    """Execute one command using `prefect_shell.ShellOperation`."""
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
