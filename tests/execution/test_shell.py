import json

import pytest

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.shell import (
    MissingPrefectShellError,
    ShellCommand,
    command_log_path,
    run_shell_command,
    run_shell_commands,
    shell_operation_kwargs,
    write_command_log_record,
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


def test_write_command_log_record_appends_backend_neutral_jsonl(tmp_path):
    pipeline_working_dir = tmp_path / "work" / "pipelines" / "image_1"
    pipeline_working_dir.mkdir(parents=True)

    log_path = write_command_log_record(
        ShellCommand(
            ["DP3", "msin=input.ms", "steps=[solve]"],
            environment={"OMP_NUM_THREADS": "4"},
            working_directory=str(pipeline_working_dir),
            name="solve",
        ),
        ExecutionConfig(),
    )

    assert log_path == tmp_path / "work" / "logs" / "commands.jsonl"
    record = json.loads(log_path.read_text().strip())
    assert record["backend"] == "prefect"
    assert record["operation"] == "image_1"
    assert record["name"] == "solve"
    assert record["command"] == ["DP3", "msin=input.ms", "steps=[solve]"]
    assert record["command_string"] == "DP3 msin=input.ms 'steps=[solve]'"
    assert record["environment"] == {"OMP_NUM_THREADS": "4"}


def test_write_command_log_record_honors_log_commands_false(tmp_path):
    pipeline_working_dir = tmp_path / "work" / "pipelines" / "image_1"
    pipeline_working_dir.mkdir(parents=True)

    log_path = write_command_log_record(
        ShellCommand(["echo", "quiet"], working_directory=str(pipeline_working_dir)),
        ExecutionConfig(log_commands=False),
    )

    assert log_path is None
    assert not (tmp_path / "work" / "logs" / "commands.jsonl").exists()


def test_command_log_path_ignores_non_operation_workdir(tmp_path):
    assert command_log_path(str(tmp_path / "not-an-operation")) is None


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
