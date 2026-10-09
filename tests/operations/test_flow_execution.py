import json
from pathlib import Path
from types import SimpleNamespace

from rapthor.execution.workspace import WORKSPACE_MARKER
from rapthor.operations.flow_execution import FlowOperation, run_prefect_flow


def test_run_prefect_flow_passes_parset_execution_config(tmp_path):
    parset = {
        "dir_working": str(tmp_path / "working"),
        "cluster_specific": {
            "debug_workflow": False,
            "keep_temporary_files": False,
            "max_cores": 4,
            "batch_system": "single_machine",
            "prefect_task_runner": "sync",
        },
    }
    payload = {"input": "value"}

    def fake_flow(payload_arg, *, execution_config):
        assert payload_arg is payload
        return {"task_runner": execution_config.task_runner}

    assert run_prefect_flow(fake_flow, payload, parset) == {"task_runner": "sync"}


def test_operation_recovers_scratch_before_recreating_its_directory(tmp_path):
    parset = {
        "dir_working": str(tmp_path / "working"),
        "cluster_specific": {
            "debug_workflow": False,
            "keep_temporary_files": False,
            "batch_system": "single_machine",
        },
    }
    workdir = Path(parset["dir_working"]) / "pipelines" / "recording_1"
    workspace = tmp_path / "scratch" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "result.txt").write_text("restart product")
    record = {"working_directory": str(workdir), "workspace": str(workspace)}
    (workspace.parent / WORKSPACE_MARKER).write_text(json.dumps(record))
    recovery_dir = workdir.parent / ".recording_1.scratch"
    recovery_dir.mkdir(parents=True)
    (recovery_dir / WORKSPACE_MARKER).write_text(json.dumps(record))

    operation = FlowOperation(SimpleNamespace(parset=parset), name="recording", index=1)

    assert Path(operation.pipeline_working_dir).resolve() == workspace
    assert (workdir / "result.txt").read_text() == "restart product"
