import json
import shlex
from pathlib import Path

import pytest
from prefect.testing.utilities import prefect_test_harness

from rapthor.execution.commands import normalize_command
from rapthor.execution.concatenate.commands import build_concatenate_command
from rapthor.execution.concatenate.flow import (
    concatenate_epoch_task,
    concatenate_flow,
)
from rapthor.execution.concatenate.payloads import concatenate_payload_from_inputs
from rapthor.execution.config import ExecutionConfig
from rapthor.lib.records import directory_record, validate_output_record
from rapthor.operations.concatenate import Concatenate
from tests.execution.conftest import run_flow_for_test

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_shell_operation_cls():
    class FakeShellOperation:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.instances.append(self)

        def run(self):
            command = self.kwargs["commands"][0]
            tokens = shlex.split(command)
            output_name = next(
                token.split("=", 1)[1] for token in tokens if token.startswith("--msout=")
            )
            cwd = Path(self.kwargs["working_dir"])
            (cwd / output_name).mkdir(parents=True, exist_ok=True)
            return "OK"

    return FakeShellOperation


class FailingShellOperation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        raise RuntimeError("concat failed")


class NoOutputShellOperation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        return "OK"


class ObservationStub:
    def __init__(self, ms_filename):
        self.ms_filename = ms_filename


class FieldStub:
    def __init__(self, parset, epoch_observations):
        self.parset = parset
        self.epoch_starttimes = [0, 1]
        self.epoch_observations = epoch_observations
        self.data_colname = "CORRECTED_DATA"
        self.ms_filenames = []
        self.scan_count = 0

    def scan_observations(self):
        self.scan_count += 1


def _operation_parset(tmp_path):
    return {
        "dir_working": str(tmp_path / "working"),
        "cluster_specific": {
            "debug_workflow": False,
            "keep_temporary_files": False,
            "max_nodes": 1,
            "batch_system": "single_machine",
            "cpus_per_task": 1,
            "mem_per_node_gb": 0,
            "dir_local": None,
            "local_scratch_dir": None,
            "global_scratch_dir": None,
            "use_container": False,
            "container_type": "docker",
            "max_cores": 1,
            "prefect_task_runner": "sync",
        },
    }


def test_build_concatenate_command_matches_reference_fixture():
    commands = json.loads((FIXTURE_DIR / "command_reference.json").read_text())

    assert (
        normalize_command(
            build_concatenate_command(
                ["epoch_0_input_0.ms", "epoch_0_input_1.ms"],
                "epoch_0_concatenated.ms",
                "DATA",
            )
        )
        == commands["concatenate"]["concat_ms_files"]
    )


def test_concatenate_payload_from_inputs_is_serializable(tmp_path):
    payload = concatenate_payload_from_inputs(
        {
            "input_filenames": [
                [
                    directory_record("/data/epoch_0_input_0.ms"),
                    directory_record("/data/epoch_0_input_1.ms"),
                ]
            ],
            "output_filenames": ["epoch_0_concatenated.ms"],
            "data_colname": "DATA",
        },
        tmp_path,
    )

    assert payload == {
        "pipeline_working_dir": str(tmp_path),
        "data_colname": "DATA",
        "epochs": [
            {
                "input_filenames": ["/data/epoch_0_input_0.ms", "/data/epoch_0_input_1.ms"],
                "output_filename": "epoch_0_concatenated.ms",
                "output_path": str(tmp_path / "epoch_0_concatenated.ms"),
            }
        ],
    }


def test_concatenate_payload_rejects_mismatched_inputs(tmp_path):
    with pytest.raises(ValueError, match="same length"):
        concatenate_payload_from_inputs(
            {
                "input_filenames": [[directory_record("/data/input.ms")]],
                "output_filenames": [],
                "data_colname": "DATA",
            },
            tmp_path,
        )


def test_concatenate_payload_rejects_non_basename_outputs(tmp_path):
    with pytest.raises(ValueError, match="basename"):
        concatenate_payload_from_inputs(
            {
                "input_filenames": [[directory_record("/data/input.ms")]],
                "output_filenames": ["nested/output.ms"],
                "data_colname": "DATA",
            },
            tmp_path,
        )


def test_concatenate_payload_rejects_duplicate_outputs(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        concatenate_payload_from_inputs(
            {
                "input_filenames": [
                    [directory_record("/data/epoch_0_input_0.ms")],
                    [directory_record("/data/epoch_1_input_0.ms")],
                ],
                "output_filenames": ["duplicated.ms", "duplicated.ms"],
                "data_colname": "DATA",
            },
            tmp_path,
        )


def test_build_concatenate_command_builds_concat_ms_tokens():
    assert build_concatenate_command(
        ["epoch_0_input_0.ms", "epoch_0_input_1.ms"],
        "epoch_0_concatenated.ms",
        "DATA",
    ) == [
        "concat_ms.py",
        "epoch_0_input_0.ms",
        "epoch_0_input_1.ms",
        "--msout=epoch_0_concatenated.ms",
        "--concat_property=frequency",
        "--data_colname=DATA",
    ]


def test_build_concatenate_command_preserves_input_order_and_data_column():
    command = build_concatenate_command(
        ["input_b.ms", "input_a.ms", "input_c.ms"],
        "ordered.ms",
        "CORRECTED_DATA",
    )

    assert command[1:4] == ["input_b.ms", "input_a.ms", "input_c.ms"]
    assert command[-1] == "--data_colname=CORRECTED_DATA"


def test_run_concatenate_flow_executes_commands_and_returns_records(
    tmp_path, fake_shell_operation_cls
):
    payload = {
        "pipeline_working_dir": str(tmp_path),
        "data_colname": "DATA",
        "epochs": [
            {
                "input_filenames": ["epoch_0_input_0.ms", "epoch_0_input_1.ms"],
                "output_filename": "epoch_0_concatenated.ms",
                "output_path": str(tmp_path / "epoch_0_concatenated.ms"),
            }
        ],
    }

    outputs = run_flow_for_test(
        concatenate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_shell_operation_cls,
    )

    assert outputs == {
        "concatenated_filenames": [directory_record(tmp_path / "epoch_0_concatenated.ms")]
    }
    validate_output_record(outputs["concatenated_filenames"])
    assert (tmp_path / "epoch_0_concatenated.ms").is_dir()
    assert fake_shell_operation_cls.instances[0].kwargs["working_dir"] == str(tmp_path)


@pytest.mark.parametrize(
    "epoch_input, expected_message",
    [
        pytest.param({}, "input_filenames", id="missing-inputs"),
        pytest.param(
            {"input_filenames": ["epoch_0_input_0.ms", 7]},
            "non-empty list of strings",
            id="malformed-inputs",
        ),
    ],
)
def test_run_concatenate_flow_rejects_invalid_epoch_input_filenames(
    tmp_path, fake_shell_operation_cls, epoch_input, expected_message
):
    payload = {
        "pipeline_working_dir": str(tmp_path),
        "data_colname": "DATA",
        "epochs": [
            {
                **epoch_input,
                "output_filename": "epoch_0_concatenated.ms",
                "output_path": str(tmp_path / "epoch_0_concatenated.ms"),
            }
        ],
    }

    with pytest.raises(ValueError, match=expected_message):
        run_flow_for_test(
            concatenate_flow,
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=fake_shell_operation_cls,
        )

    assert fake_shell_operation_cls.instances == []


def test_concatenate_epoch_task_wraps_epoch_runner(tmp_path, fake_shell_operation_cls):
    epoch = {
        "input_filenames": ["epoch_0_input_0.ms", "epoch_0_input_1.ms"],
        "output_filename": "epoch_0_concatenated.ms",
        "output_path": str(tmp_path / "epoch_0_concatenated.ms"),
    }

    task_fn = getattr(concatenate_epoch_task, "fn", concatenate_epoch_task)
    output = task_fn(
        epoch,
        "DATA",
        str(tmp_path),
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_shell_operation_cls,
    )

    assert output == directory_record(tmp_path / "epoch_0_concatenated.ms")
    assert fake_shell_operation_cls.instances[0].kwargs["working_dir"] == str(tmp_path)


def test_concatenate_prefect_flow_entrypoint_runs_with_mocked_shell(
    tmp_path, monkeypatch, fake_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_shell_operation_cls,
    )
    payload = {
        "pipeline_working_dir": str(tmp_path),
        "data_colname": "DATA",
        "epochs": [
            {
                "input_filenames": ["epoch_0_input_0.ms", "epoch_0_input_1.ms"],
                "output_filename": "epoch_0_concatenated.ms",
                "output_path": str(tmp_path / "epoch_0_concatenated.ms"),
            }
        ],
    }

    with prefect_test_harness(server_startup_timeout=None):
        outputs = concatenate_flow(
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
        )

    assert outputs == {
        "concatenated_filenames": [directory_record(tmp_path / "epoch_0_concatenated.ms")]
    }
    assert len(fake_shell_operation_cls.instances) == 1
    assert fake_shell_operation_cls.instances[0].kwargs == {
        "commands": [
            "concat_ms.py epoch_0_input_0.ms epoch_0_input_1.ms "
            "--msout=epoch_0_concatenated.ms --concat_property=frequency "
            "--data_colname=DATA"
        ],
        "working_dir": str(tmp_path),
        "stream_output": True,
    }


def test_run_concatenate_flow_handles_no_concatenation_needed(tmp_path, fake_shell_operation_cls):
    outputs = run_flow_for_test(
        concatenate_flow,
        {"pipeline_working_dir": str(tmp_path), "data_colname": "DATA", "epochs": []},
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_shell_operation_cls,
    )

    assert outputs == {"concatenated_filenames": []}
    assert fake_shell_operation_cls.instances == []


def test_run_concatenate_flow_propagates_shell_failures(tmp_path):
    payload = {
        "pipeline_working_dir": str(tmp_path),
        "data_colname": "DATA",
        "epochs": [
            {
                "input_filenames": ["epoch_0_input_0.ms", "epoch_0_input_1.ms"],
                "output_filename": "epoch_0_concatenated.ms",
                "output_path": str(tmp_path / "epoch_0_concatenated.ms"),
            }
        ],
    }

    with pytest.raises(RuntimeError, match="concat failed"):
        run_flow_for_test(
            concatenate_flow,
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=FailingShellOperation,
        )

    assert not (tmp_path / "epoch_0_concatenated.ms").exists()


def test_run_concatenate_flow_fails_when_expected_output_is_missing(tmp_path):
    payload = {
        "pipeline_working_dir": str(tmp_path),
        "data_colname": "DATA",
        "epochs": [
            {
                "input_filenames": ["epoch_0_input_0.ms", "epoch_0_input_1.ms"],
                "output_filename": "epoch_0_concatenated.ms",
                "output_path": str(tmp_path / "epoch_0_concatenated.ms"),
            }
        ],
    }

    with pytest.raises(FileNotFoundError, match="Concatenate output was not created"):
        run_flow_for_test(
            concatenate_flow,
            payload,
            execution_config=ExecutionConfig(task_runner="sync"),
            shell_operation_cls=NoOutputShellOperation,
        )


def test_concatenate_finalizer_accepts_prefect_outputs(tmp_path, fake_shell_operation_cls):
    field = FieldStub(
        _operation_parset(tmp_path),
        epoch_observations=[
            [ObservationStub("epoch_0_input_0.ms"), ObservationStub("epoch_0_input_1.ms")],
            [ObservationStub("epoch_1_single.ms")],
        ],
    )
    operation = Concatenate(field, index=0)
    operation.set_input_parameters()
    payload = concatenate_payload_from_inputs(operation.input_parms, operation.pipeline_working_dir)
    outputs = run_flow_for_test(
        concatenate_flow,
        payload,
        execution_config=ExecutionConfig(task_runner="sync"),
        shell_operation_cls=fake_shell_operation_cls,
    )

    operation.outputs = outputs
    operation.finalize()

    assert field.ms_filenames == [
        str(Path(operation.pipeline_working_dir) / "epoch_0_concatenated.ms"),
        "epoch_1_single.ms",
    ]
    assert field.data_colname == "DATA"
    assert field.scan_count == 1
    assert Path(operation.done_file).is_file()


def test_concatenate_operation_run_uses_prefect_flow(
    tmp_path, monkeypatch, fake_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_shell_operation_cls,
    )
    field = FieldStub(
        _operation_parset(tmp_path),
        epoch_observations=[
            [ObservationStub("epoch_0_input_0.ms"), ObservationStub("epoch_0_input_1.ms")],
            [ObservationStub("epoch_1_single.ms")],
        ],
    )
    operation = Concatenate(field, index=0)

    with prefect_test_harness(server_startup_timeout=None):
        operation.run()

    expected_output = directory_record(
        Path(operation.pipeline_working_dir) / "epoch_0_concatenated.ms"
    )
    assert operation.outputs == {"concatenated_filenames": [expected_output]}
    assert json.loads(Path(operation.outputs_file).read_text()) == {
        "concatenated_filenames": [expected_output]
    }
    assert Path(operation.done_file).is_file()
    assert Path(operation.pipeline_inputs_file).is_file()
    assert field.ms_filenames == [
        str(Path(operation.pipeline_working_dir) / "epoch_0_concatenated.ms"),
        "epoch_1_single.ms",
    ]
    assert field.scan_count == 1
    assert len(fake_shell_operation_cls.instances) == 1


def test_concatenate_operation_run_reuses_prefect_outputs_when_done(
    tmp_path, monkeypatch, fake_shell_operation_cls
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: fake_shell_operation_cls,
    )
    field = FieldStub(
        _operation_parset(tmp_path),
        epoch_observations=[
            [ObservationStub("epoch_0_input_0.ms"), ObservationStub("epoch_0_input_1.ms")],
            [ObservationStub("epoch_1_single.ms")],
        ],
    )
    operation = Concatenate(field, index=0)
    expected_output = directory_record(
        Path(operation.pipeline_working_dir) / "epoch_0_concatenated.ms"
    )
    Path(operation.done_file).touch()
    Path(operation.outputs_file).write_text(
        json.dumps({"concatenated_filenames": [expected_output]})
    )

    operation.run()

    assert operation.outputs == {"concatenated_filenames": [expected_output]}
    assert Path(operation.done_file).is_file()
    assert field.ms_filenames == [
        str(Path(operation.pipeline_working_dir) / "epoch_0_concatenated.ms"),
        "epoch_1_single.ms",
    ]
    assert field.scan_count == 1
    assert fake_shell_operation_cls.instances == []


@pytest.mark.parametrize(
    "shell_operation_cls, expected_message",
    [
        pytest.param(FailingShellOperation, "concat failed", id="shell-failure"),
        pytest.param(
            NoOutputShellOperation,
            "Concatenate output was not created",
            id="missing-concatenated-output",
        ),
    ],
)
def test_concatenate_operation_run_failure_does_not_mark_done(
    tmp_path, monkeypatch, shell_operation_cls, expected_message
):
    monkeypatch.setattr(
        "rapthor.execution.shell._load_shell_operation_cls",
        lambda: shell_operation_cls,
    )
    field = FieldStub(
        _operation_parset(tmp_path),
        epoch_observations=[
            [ObservationStub("epoch_0_input_0.ms"), ObservationStub("epoch_0_input_1.ms")],
            [ObservationStub("epoch_1_single.ms")],
        ],
    )
    operation = Concatenate(field, index=0)

    with (
        prefect_test_harness(server_startup_timeout=None),
        pytest.raises((FileNotFoundError, RuntimeError), match=expected_message),
    ):
        operation.run()

    assert Path(operation.pipeline_inputs_file).is_file()
    assert not Path(operation.done_file).exists()
    assert not Path(operation.outputs_file).exists()
    assert operation.outputs == {}
    assert field.ms_filenames == []
    assert field.data_colname == "CORRECTED_DATA"
    assert field.scan_count == 0
