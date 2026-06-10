"""Shell command wrappers used by Prefect tasks."""

import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

from rapthor.execution.commands import CommandInput, command_to_string, normalize_command
from rapthor.execution.config import ExecutionConfig


class MissingPrefectShellError(RuntimeError):
    """Raised when prefect-shell is required but not installed."""


log = logging.getLogger("rapthor:shell")
STREAM_LOG_FLUSH_INTERVAL_SECONDS = 1.0
STREAM_LOG_MAX_LINES_PER_RECORD = 40
_STREAM_DONE = object()


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


def _is_prefect_shell_operation_cls(operation_cls) -> bool:
    return (
        getattr(operation_cls, "__module__", None) == "prefect_shell.commands"
        and getattr(operation_cls, "__name__", None) == "ShellOperation"
    )


def _get_command_logger():
    try:
        from prefect.logging import get_run_logger

        return get_run_logger()
    except Exception:
        return log


def _queue_process_output(output, output_queue) -> None:
    try:
        for raw_line in iter(output.readline, b""):
            output_queue.put(raw_line)
    finally:
        output_queue.put(_STREAM_DONE)


def _output_lines(raw_line: bytes) -> list[str]:
    text = raw_line.decode(errors="replace").rstrip("\r\n")
    return text.splitlines() or [text]


def _flush_stream_log(command_logger, buffered_lines: list[str]) -> None:
    if not buffered_lines:
        return
    message = "\n".join(buffered_lines).rstrip("\n")
    buffered_lines.clear()
    if message:
        command_logger.info(message)


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


def _run_streaming_shell_command(shell_command: ShellCommand) -> list[str]:
    command_logger = _get_command_logger()
    buffered_log_lines = []
    output_lines = []
    process = None
    reader_thread = None
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(
            prefix="rapthor-prefect-",
            suffix=".sh",
            delete=False,
        )
        temp_file.write(shell_command.command_string.encode())
        temp_file.close()

        env = os.environ.copy()
        env.update(shell_command.environment)
        process = subprocess.Popen(
            ["bash", temp_file.name],
            cwd=shell_command.working_directory,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if process.stdout is None:
            raise RuntimeError("Command output pipe was not created")

        output_queue = queue.Queue()
        reader_thread = threading.Thread(
            target=_queue_process_output,
            args=(process.stdout, output_queue),
            daemon=True,
        )
        reader_thread.start()

        while True:
            try:
                raw_line = output_queue.get(timeout=STREAM_LOG_FLUSH_INTERVAL_SECONDS)
            except queue.Empty:
                _flush_stream_log(command_logger, buffered_log_lines)
                continue

            if raw_line is _STREAM_DONE:
                break

            lines = _output_lines(raw_line)
            output_lines.extend(lines)
            buffered_log_lines.extend(lines)
            if len(buffered_log_lines) >= STREAM_LOG_MAX_LINES_PER_RECORD:
                _flush_stream_log(command_logger, buffered_log_lines)

        _flush_stream_log(command_logger, buffered_log_lines)

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(
                "Command failed with return code "
                f"{process.returncode}: {shell_command.command_string}"
            )
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if reader_thread is not None:
            reader_thread.join(timeout=1)
        if temp_file is not None and os.path.exists(temp_file.name):
            os.remove(temp_file.name)

    return output_lines


def run_shell_command(
    shell_command: ShellCommand,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
):
    """Execute one command using the configured Prefect shell runner."""
    write_command_log_record(shell_command, execution_config)
    operation_cls = shell_operation_cls or _load_shell_operation_cls()
    if (
        shell_operation_cls is None
        and execution_config.stream_output
        and _is_prefect_shell_operation_cls(operation_cls)
    ):
        return _run_streaming_shell_command(shell_command)

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
