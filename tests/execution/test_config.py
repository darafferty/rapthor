import pytest

from rapthor.execution.config import ExecutionConfig


def test_execution_config_defaults_from_empty_parset():
    config = ExecutionConfig.from_parset({})

    assert config.task_runner == "local_dask"
    assert config.dask_scheduler is None
    assert config.stream_output is True
    assert config.retries == 0
    assert config.log_commands is True
    assert config.batch_system == "single_machine"
    assert config.effective_local_scratch_dir is None


def test_execution_config_reads_cluster_specific_values():
    config = ExecutionConfig.from_parset(
        {
            "cluster_specific": {
                "prefect_task_runner": "sync",
                "dask_scheduler": "tcp://scheduler:8786",
                "prefect_stream_output": False,
                "prefect_retries": 2,
                "prefect_log_commands": False,
                "batch_system": "slurm",
                "max_nodes": 4,
                "cpus_per_task": 32,
                "mem_per_node_gb": 256,
                "use_container": True,
                "container_type": "singularity",
                "local_scratch_dir": "/local",
                "global_scratch_dir": "/shared",
                "dir_local": "/deprecated",
            }
        }
    )

    assert config.task_runner == "sync"
    assert config.dask_scheduler == "tcp://scheduler:8786"
    assert config.stream_output is False
    assert config.retries == 2
    assert config.log_commands is False
    assert config.batch_system == "slurm"
    assert config.max_nodes == 4
    assert config.cpus_per_task == 32
    assert config.mem_per_node_gb == 256
    assert config.use_container is True
    assert config.container_type == "singularity"
    assert config.effective_local_scratch_dir == "/local"
    assert config.global_scratch_dir == "/shared"


def test_execution_config_infers_external_dask_from_scheduler():
    config = ExecutionConfig.from_parset(
        {"cluster_specific": {"dask_scheduler": "tcp://scheduler:8786"}}
    )

    assert config.task_runner == "external_dask"


def test_execution_config_uses_deprecated_dir_local_as_scratch_fallback():
    config = ExecutionConfig.from_parset({"cluster_specific": {"dir_local": "/tmp/rapthor"}})

    assert config.effective_local_scratch_dir == "/tmp/rapthor"


def test_execution_config_rejects_invalid_task_runner():
    with pytest.raises(ValueError, match="prefect_task_runner"):
        ExecutionConfig.from_parset({"cluster_specific": {"prefect_task_runner": "toil"}})


def test_execution_config_rejects_negative_retries():
    with pytest.raises(ValueError, match="prefect_retries"):
        ExecutionConfig.from_parset({"cluster_specific": {"prefect_retries": -1}})
