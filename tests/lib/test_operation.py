"""
Test cases for the `rapthor.lib.operation` module.
"""

import json
from pathlib import Path

import pytest

from rapthor.lib.operation import Operation


def _operation_parset(tmp_path):
    return {
        "dir_working": str(tmp_path / "working"),
        "cluster_specific": {
            "cwl_runner": "mock_cwl_runner",
            "debug_workflow": False,
            "keep_temporary_files": False,
            "max_nodes": 1,
            "max_cores": 4,
            "batch_system": "single_machine",
            "cpus_per_task": 1,
            "mem_per_node_gb": 1,
            "dir_local": str(tmp_path / "scratch"),
            "local_scratch_dir": str(tmp_path / "local_scratch"),
            "global_scratch_dir": str(tmp_path / "global_scratch"),
            "use_container": False,
        },
    }


class FieldStub:
    def __init__(self, parset):
        self.parset = parset


class RecordingOperation(Operation):
    def __init__(
        self,
        field,
        *,
        index=1,
        workflow_success=True,
        workflow_outputs=None,
        fail_finalize=False,
    ):
        super().__init__(field, index=index, name="Recording")
        self.workflow_success = workflow_success
        self.workflow_outputs = workflow_outputs or {
            "result": {"class": "File", "path": str(Path(self.pipeline_working_dir) / "result.txt")}
        }
        self.fail_finalize = fail_finalize
        self.calls = []

    def set_parset_parameters(self):
        self.calls.append("set_parset_parameters")
        self.parset_parms = {"unused": "value"}

    def set_input_parameters(self):
        self.calls.append("set_input_parameters")
        self.input_parms = {"input": "value"}

    def execute_workflow(self):
        self.calls.append("execute_workflow")
        return self.workflow_success, self.workflow_outputs

    def finalize(self):
        self.calls.append("finalize")
        if self.fail_finalize:
            raise RuntimeError("finalizer failed")
        super().finalize()


def _operation(tmp_path, **kwargs):
    return RecordingOperation(FieldStub(_operation_parset(tmp_path)), **kwargs)


def test_setup_writes_python_flow_inputs_without_rendering_cwl(tmp_path):
    operation = _operation(tmp_path)

    operation.setup()

    assert operation.calls == ["set_parset_parameters", "set_input_parameters"]
    assert json.loads(Path(operation.pipeline_inputs_file).read_text()) == {"input": "value"}
    assert not Path(operation.pipeline_parset_file).exists()


def test_base_operation_requires_prefect_execute_workflow(tmp_path):
    operation = Operation(FieldStub(_operation_parset(tmp_path)), name="Base")

    with pytest.raises(NotImplementedError, match="CWL execution has been retired"):
        operation.run()

    assert json.loads(Path(operation.pipeline_inputs_file).read_text()) == {}
    assert not Path(operation.pipeline_parset_file).exists()


def test_flow_max_cores_uses_cluster_hint_for_non_slurm(tmp_path):
    operation = Operation(FieldStub(_operation_parset(tmp_path)), name="Base")

    assert operation.flow_max_cores() == 4


def test_flow_max_cores_omits_hint_for_slurm(tmp_path):
    parset = _operation_parset(tmp_path)
    parset["cluster_specific"]["batch_system"] = "slurm"
    operation = Operation(FieldStub(parset), name="Base")

    assert operation.flow_max_cores() is None


def test_run_prefect_flow_passes_parset_execution_config(tmp_path):
    parset = _operation_parset(tmp_path)
    parset["cluster_specific"]["prefect_task_runner"] = "sync"
    operation = Operation(FieldStub(parset), name="Base")
    payload = {"input": "value"}

    def fake_flow(payload_arg, *, execution_config):
        assert payload_arg is payload
        return {"task_runner": execution_config.task_runner}

    assert operation.run_prefect_flow(fake_flow, payload) == {"task_runner": "sync"}


def test_finalize_marks_operation_done(tmp_path):
    operation = _operation(tmp_path)

    operation.finalize()

    assert Path(operation.done_file).is_file()


def test_run_executes_workflow_stores_outputs_and_marks_done(tmp_path):
    operation = _operation(tmp_path)

    operation.run()

    assert operation.calls == [
        "set_parset_parameters",
        "set_input_parameters",
        "execute_workflow",
        "finalize",
    ]
    assert Path(operation.done_file).is_file()
    assert json.loads(Path(operation.outputs_file).read_text()) == operation.workflow_outputs
    assert operation.outputs == operation.workflow_outputs


def test_run_reuses_outputs_when_done(tmp_path):
    operation = _operation(tmp_path)
    persisted_outputs = {"persisted": {"class": "Directory", "path": "already-done.ms"}}
    Path(operation.done_file).touch()
    Path(operation.outputs_file).write_text(json.dumps(persisted_outputs))

    operation.run()

    assert operation.calls == [
        "set_parset_parameters",
        "set_input_parameters",
        "finalize",
    ]
    assert operation.outputs == persisted_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == persisted_outputs
    assert Path(operation.done_file).is_file()


def test_run_reruns_workflow_after_done_marker_is_deleted(tmp_path):
    operation = _operation(tmp_path)
    stale_outputs = {"persisted": {"class": "Directory", "path": "old.ms"}}
    Path(operation.outputs_file).write_text(json.dumps(stale_outputs))

    operation.run()

    assert "execute_workflow" in operation.calls
    assert operation.outputs == operation.workflow_outputs
    assert json.loads(Path(operation.outputs_file).read_text()) == operation.workflow_outputs
    assert Path(operation.done_file).is_file()


def test_run_with_done_marker_fails_when_outputs_file_is_missing(tmp_path):
    operation = _operation(tmp_path)
    Path(operation.done_file).touch()

    with pytest.raises(RuntimeError, match="marked done but outputs file .* is missing"):
        operation.run()

    assert operation.calls == ["set_parset_parameters", "set_input_parameters"]


def test_run_with_done_marker_fails_when_outputs_file_is_corrupt(tmp_path):
    operation = _operation(tmp_path)
    Path(operation.done_file).touch()
    Path(operation.outputs_file).write_text("{")

    with pytest.raises(RuntimeError, match="outputs file .* is not valid JSON"):
        operation.run()

    assert operation.calls == ["set_parset_parameters", "set_input_parameters"]


def test_failed_workflow_does_not_mark_operation_done_or_store_outputs(tmp_path):
    operation = _operation(tmp_path, workflow_success=False)

    with pytest.raises(RuntimeError, match="Operation recording_1 failed due to an error"):
        operation.run()

    assert operation.calls == [
        "set_parset_parameters",
        "set_input_parameters",
        "execute_workflow",
    ]
    assert not Path(operation.done_file).exists()
    assert not Path(operation.outputs_file).exists()


def test_finalizer_failure_does_not_mark_operation_done_or_store_outputs(tmp_path):
    operation = _operation(tmp_path, fail_finalize=True)

    with pytest.raises(RuntimeError, match="finalizer failed"):
        operation.run()

    assert operation.calls == [
        "set_parset_parameters",
        "set_input_parameters",
        "execute_workflow",
        "finalize",
    ]
    assert not Path(operation.done_file).exists()
    assert not Path(operation.outputs_file).exists()
