import pytest

from rapthor.execution.config import (
    DASK_SCHEDULER_ENV,
    PREFECT_API_URL_ENV,
    ExecutionConfig,
    dask_scheduler_from_environment,
    prefect_api_url_from_environment,
)


def test_execution_config_defaults_from_empty_parset(monkeypatch):
    monkeypatch.delenv(DASK_SCHEDULER_ENV, raising=False)
    monkeypatch.delenv(PREFECT_API_URL_ENV, raising=False)

    config = ExecutionConfig.from_parset({})

    assert config.task_runner == "local_dask"
    assert config.prefect_api_mode == "auto"
    assert config.prefect_api_url is None
    assert config.dask_scheduler is None
    assert config.dask_dashboard_address is None
    assert config.stream_output is True
    assert config.retries == 0
    assert config.log_commands is True
    assert config.command_profile == "auto"
    assert config.publish_fits_previews is False
    assert config.batch_system == "single_machine"
    assert config.local_scratch_dir is None


def test_execution_config_reads_cluster_specific_values():
    config = ExecutionConfig.from_parset(
        {
            "cluster_specific": {
                "prefect_task_runner": "sync",
                "prefect_api_mode": "external",
                "prefect_api_url": "http://prefect:4200/api",
                "dask_scheduler": "tcp://scheduler:8786",
                "dask_dashboard_address": ":8787",
                "prefect_stream_output": False,
                "prefect_retries": 2,
                "prefect_log_commands": False,
                "prefect_command_profile": "time",
                "prefect_publish_fits_previews": True,
                "batch_system": "slurm",
                "max_nodes": 4,
                "local_dask_workers": 3,
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
    assert config.prefect_api_mode == "external"
    assert config.prefect_api_url == "http://prefect:4200/api"
    assert config.dask_scheduler == "tcp://scheduler:8786"
    assert config.dask_dashboard_address == ":8787"
    assert config.stream_output is False
    assert config.retries == 2
    assert config.log_commands is False
    assert config.command_profile == "time"
    assert config.publish_fits_previews is True
    assert config.batch_system == "slurm"
    assert config.max_nodes == 4
    assert config.local_dask_workers == 3
    assert config.cpus_per_task == 32
    assert config.mem_per_node_gb == 256
    assert config.use_container is True
    assert config.container_type == "singularity"
    assert config.local_scratch_dir == "/local"
    assert config.global_scratch_dir == "/shared"
    assert config.deprecated_dir_local == "/deprecated"


def test_execution_config_infers_external_dask_from_scheduler():
    config = ExecutionConfig.from_parset(
        {"cluster_specific": {"dask_scheduler": "tcp://scheduler:8786"}}
    )

    assert config.task_runner == "external_dask"


def test_dask_scheduler_from_environment():
    scheduler = dask_scheduler_from_environment({DASK_SCHEDULER_ENV: "tcp://env:8786"})

    assert scheduler == "tcp://env:8786"


def test_prefect_api_url_from_environment():
    api_url = prefect_api_url_from_environment(
        {PREFECT_API_URL_ENV: "http://prefect.example:4200/api"}
    )

    assert api_url == "http://prefect.example:4200/api"


def test_execution_config_infers_external_dask_from_environment(monkeypatch):
    monkeypatch.setenv(DASK_SCHEDULER_ENV, "tcp://env:8786")

    config = ExecutionConfig.from_parset({})

    assert config.task_runner == "external_dask"
    assert config.dask_scheduler == "tcp://env:8786"


@pytest.mark.parametrize("task_runner", ["sync", "local_dask", "external_dask"])
def test_execution_config_explicit_task_runner_overrides_auto_selection(monkeypatch, task_runner):
    monkeypatch.setenv(DASK_SCHEDULER_ENV, "tcp://env:8786")

    config = ExecutionConfig.from_parset({"cluster_specific": {"prefect_task_runner": task_runner}})

    assert config.task_runner == task_runner
    assert config.dask_scheduler == "tcp://env:8786"


def test_execution_config_prefers_parset_scheduler_over_environment(monkeypatch):
    monkeypatch.setenv(DASK_SCHEDULER_ENV, "tcp://env:8786")

    config = ExecutionConfig.from_parset(
        {"cluster_specific": {"dask_scheduler": "tcp://parset:8786"}}
    )

    assert config.dask_scheduler == "tcp://parset:8786"


def test_execution_config_reads_prefect_api_url_from_environment(monkeypatch):
    monkeypatch.setenv(PREFECT_API_URL_ENV, "http://prefect-env:4200/api")

    config = ExecutionConfig.from_parset({})

    assert config.prefect_api_url == "http://prefect-env:4200/api"


def test_execution_config_prefers_parset_prefect_api_url_over_environment(monkeypatch):
    monkeypatch.setenv(PREFECT_API_URL_ENV, "http://prefect-env:4200/api")

    config = ExecutionConfig.from_parset(
        {"cluster_specific": {"prefect_api_url": "http://prefect-parset:4200/api"}}
    )

    assert config.prefect_api_url == "http://prefect-parset:4200/api"


def test_execution_config_uses_deprecated_dir_local_as_scratch_fallback():
    config = ExecutionConfig.from_parset({"cluster_specific": {"dir_local": "/tmp/rapthor"}})

    assert config.deprecated_dir_local == "/tmp/rapthor"


def test_execution_config_exposes_effective_local_dask_capacity():
    config = ExecutionConfig(max_nodes=0, cpus_per_task=0)

    assert config.local_dask_worker_count == 1
    assert config.local_dask_threads_per_worker == 1

    config = ExecutionConfig(max_nodes=4, local_dask_workers=0, cpus_per_task=8)

    assert config.local_dask_worker_count == 4
    assert config.local_dask_threads_per_worker == 8

    config = ExecutionConfig(max_nodes=1, local_dask_workers=4, cpus_per_task=8)

    assert config.local_dask_worker_count == 4
    assert config.local_dask_threads_per_worker == 8


def test_execution_config_rejects_invalid_task_runner():
    with pytest.raises(ValueError, match="prefect_task_runner"):
        ExecutionConfig.from_parset({"cluster_specific": {"prefect_task_runner": "toil"}})


def test_execution_config_rejects_invalid_prefect_api_mode():
    with pytest.raises(ValueError, match="prefect_api_mode"):
        ExecutionConfig.from_parset({"cluster_specific": {"prefect_api_mode": "always"}})


def test_execution_config_rejects_invalid_command_profile():
    with pytest.raises(ValueError, match="prefect_command_profile"):
        ExecutionConfig.from_parset({"cluster_specific": {"prefect_command_profile": "always"}})


def test_execution_config_treats_none_command_profile_as_auto():
    config = ExecutionConfig.from_parset({"cluster_specific": {"prefect_command_profile": None}})

    assert config.command_profile == "auto"


def test_execution_config_rejects_negative_retries():
    with pytest.raises(ValueError, match="prefect_retries"):
        ExecutionConfig.from_parset({"cluster_specific": {"prefect_retries": -1}})


def test_execution_config_rejects_negative_local_dask_workers():
    with pytest.raises(ValueError, match="local_dask_workers"):
        ExecutionConfig.from_parset({"cluster_specific": {"local_dask_workers": -1}})


def test_execution_config_rejects_invalid_fits_preview_artifact_flag():
    with pytest.raises(ValueError, match="prefect_publish_fits_previews"):
        ExecutionConfig.from_parset(
            {"cluster_specific": {"prefect_publish_fits_previews": "sometimes"}}
        )
