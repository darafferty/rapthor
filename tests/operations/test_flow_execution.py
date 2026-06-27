from rapthor.operations.flow_execution import run_prefect_flow


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
