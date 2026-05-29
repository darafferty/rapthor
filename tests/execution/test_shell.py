import pytest

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.shell import (
    MissingPrefectShellError,
    ShellCommand,
    run_shell_command,
    run_shell_commands,
    shell_operation_kwargs,
)


class FakeShellOperation:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)

    def run(self):
        return "OK"


def test_shell_command_formats_tokens():
    command = ShellCommand(["echo", "hello world"])

    assert command.command_string == "echo 'hello world'"


def test_shell_operation_kwargs_include_env_and_cwd():
    kwargs = shell_operation_kwargs(
        ShellCommand(
            "echo hello",
            environment={"OPENBLAS_NUM_THREADS": "1"},
            working_directory="/tmp/task",
        ),
        ExecutionConfig(stream_output=False),
    )

    assert kwargs == {
        "commands": ["echo hello"],
        "stream_output": False,
        "env": {"OPENBLAS_NUM_THREADS": "1"},
        "cwd": "/tmp/task",
    }


def test_run_shell_command_uses_injected_operation_class():
    FakeShellOperation.instances = []

    result = run_shell_command(
        ShellCommand(["echo", "hello"]),
        ExecutionConfig(),
        shell_operation_cls=FakeShellOperation,
    )

    assert result == "OK"
    assert FakeShellOperation.instances[0].kwargs["commands"] == ["echo hello"]


def test_run_shell_commands_runs_sequentially():
    FakeShellOperation.instances = []

    result = run_shell_commands(
        [ShellCommand(["echo", "one"]), ShellCommand(["echo", "two"])],
        ExecutionConfig(),
        shell_operation_cls=FakeShellOperation,
    )

    assert result == ["OK", "OK"]
    assert [item.kwargs["commands"][0] for item in FakeShellOperation.instances] == [
        "echo one",
        "echo two",
    ]


def test_run_shell_command_requires_prefect_shell_without_injection(monkeypatch):
    def missing_shell_operation():
        raise MissingPrefectShellError("prefect-shell is required")

    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        missing_shell_operation,
    )

    with pytest.raises(MissingPrefectShellError, match="prefect-shell"):
        run_shell_command(ShellCommand(["echo", "hello"]), ExecutionConfig())
