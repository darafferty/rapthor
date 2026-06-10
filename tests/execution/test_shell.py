import json
import logging
import sys

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


def test_shell_operation_kwargs_include_env_and_working_dir():
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
        "working_dir": "/tmp/task",
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


def test_run_shell_command_records_duration_metadata(tmp_path):
    FakeShellOperation.instances = []
    pipeline_working_dir = tmp_path / "work" / "pipelines" / "calibrate_1"
    pipeline_working_dir.mkdir(parents=True)

    result = run_shell_command(
        ShellCommand(
            ["DP3", "msin=input.ms"],
            working_directory=str(pipeline_working_dir),
            name="solve",
        ),
        ExecutionConfig(),
        shell_operation_cls=FakeShellOperation,
    )

    assert result == "OK"
    log_path = tmp_path / "work" / "logs" / "commands.jsonl"
    record = json.loads(log_path.read_text())
    assert record["operation"] == "calibrate_1"
    assert record["name"] == "solve"
    assert record["status"] == "completed"
    assert record["returncode"] == 0
    assert record["duration_seconds"] >= 0
    assert record["started_at"]
    assert record["finished_at"]


def test_run_shell_command_streams_clean_output_to_logger(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="rapthor:shell")

    result = run_shell_command(
        ShellCommand(
            [sys.executable, "-c", "print('clean line')"],
            working_directory=str(tmp_path),
        ),
        ExecutionConfig(stream_output=True),
    )

    assert result == ["clean line"]
    assert "clean line" in caplog.text
    assert "PID" not in caplog.text
    assert "stream output" not in caplog.text


def test_run_shell_command_batches_nearby_output_lines(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="rapthor:shell")

    result = run_shell_command(
        ShellCommand(
            [sys.executable, "-c", "print('first'); print('second'); print('third')"],
            working_directory=str(tmp_path),
        ),
        ExecutionConfig(stream_output=True),
    )

    messages = [record.getMessage() for record in caplog.records if record.name == "rapthor:shell"]
    assert result == ["first", "second", "third"]
    assert len(messages) == 1
    assert "first\nsecond\nthird" in messages[0]
    assert "first" not in messages
    assert "second" not in messages
    assert "third" not in messages


def test_run_shell_command_streaming_raises_on_failure(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="rapthor:shell")
    pipeline_working_dir = tmp_path / "work" / "pipelines" / "calibrate_1"
    pipeline_working_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="return code 7"):
        run_shell_command(
            ShellCommand(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('failure line', file=sys.stderr); sys.exit(7)",
                ],
                working_directory=str(pipeline_working_dir),
                name="failing-step",
            ),
            ExecutionConfig(stream_output=True),
        )

    assert "failure line" in caplog.text
    assert "PID" not in caplog.text
    log_path = tmp_path / "work" / "logs" / "commands.jsonl"
    record = json.loads(log_path.read_text())
    assert record["operation"] == "calibrate_1"
    assert record["name"] == "failing-step"
    assert record["status"] == "failed"
    assert record["returncode"] == 7
    assert "return code 7" in record["error"]


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
