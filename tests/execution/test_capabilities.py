import pytest

from rapthor.execution.capabilities import (
    PreflightError,
    collect_preflight_issues,
    preflight_execution,
)
from rapthor.execution.config import DASK_SCHEDULER_ENV, ExecutionConfig
from rapthor.execution.resources import ResourceRequest


def test_preflight_passes_for_default_local_config():
    preflight_execution(ExecutionConfig())


def test_preflight_reports_missing_external_dask_scheduler(monkeypatch):
    monkeypatch.delenv(DASK_SCHEDULER_ENV, raising=False)
    config = ExecutionConfig(task_runner="external_dask")

    issues = collect_preflight_issues(config)

    assert [issue.code for issue in issues] == ["missing_dask_scheduler"]
    assert issues[0].option == "dask_scheduler"


def test_preflight_accepts_environment_external_dask_scheduler(monkeypatch):
    monkeypatch.setenv(DASK_SCHEDULER_ENV, "tcp://scheduler:8786")
    config = ExecutionConfig(task_runner="external_dask")

    issues = collect_preflight_issues(config)

    assert issues == []


def test_preflight_can_check_external_dask_scheduler():
    config = ExecutionConfig(task_runner="external_dask", dask_scheduler="tcp://scheduler:8786")
    checked = []

    issues = collect_preflight_issues(config, scheduler_checker=checked.append)

    assert issues == []
    assert checked == ["tcp://scheduler:8786"]


def test_preflight_reports_failed_external_dask_scheduler_check():
    config = ExecutionConfig(task_runner="external_dask", dask_scheduler="tcp://scheduler:8786")

    def fail(address):
        raise RuntimeError(f"{address} is down")

    issues = collect_preflight_issues(config, scheduler_checker=fail)

    assert [issue.code for issue in issues] == ["dask_scheduler_unreachable"]
    assert issues[0].option == "dask_scheduler"
    assert "is down" in issues[0].message


def test_preflight_reports_slurm_requires_external_dask():
    config = ExecutionConfig(task_runner="local_dask", batch_system="slurm")

    issues = collect_preflight_issues(config)

    assert [issue.code for issue in issues] == ["slurm_requires_external_dask"]
    assert issues[0].option == "slurm"


def test_preflight_reports_unsupported_container_mode():
    config = ExecutionConfig(use_container=True, container_type="singularity")

    with pytest.raises(PreflightError) as exc:
        preflight_execution(config)

    assert exc.value.issues[0].code == "unsupported_container"


def test_preflight_reports_unsupported_features():
    config = ExecutionConfig()

    with pytest.raises(PreflightError) as exc:
        preflight_execution(
            config,
            requested_features={"concatenate", "mpi_wsclean"},
            supported_features={"concatenate"},
        )

    assert exc.value.issues[0].code == "unsupported_feature"
    assert "mpi_wsclean" in exc.value.issues[0].message


def test_preflight_reports_missing_tools():
    config = ExecutionConfig()

    with pytest.raises(PreflightError) as exc:
        preflight_execution(
            config,
            required_tools=["DP3", "wsclean"],
            tool_resolver=lambda tool: "/usr/bin/DP3" if tool == "DP3" else None,
        )

    assert [issue.code for issue in exc.value.issues] == ["missing_tool"]
    assert "wsclean" in exc.value.issues[0].message


def test_preflight_reports_invalid_resource_requests():
    config = ExecutionConfig(cpus_per_task=4, max_nodes=1)

    with pytest.raises(PreflightError) as exc:
        preflight_execution(
            config,
            resource_requests=[
                ResourceRequest(name="dp3", threads=8),
                ResourceRequest(
                    name="wsclean-mpi",
                    processes=2,
                    use_mpi=True,
                    exclusive=False,
                ),
            ],
        )

    assert [issue.code for issue in exc.value.issues] == [
        "resource_threads_oversubscribed",
        "mpi_not_exclusive",
        "mpi_processes_oversubscribed",
    ]
    assert all(issue.option == "resources" for issue in exc.value.issues)


def test_preflight_reports_local_dask_process_oversubscription():
    config = ExecutionConfig(task_runner="local_dask", max_nodes=1)

    with pytest.raises(PreflightError) as exc:
        preflight_execution(
            config,
            resource_requests=[ResourceRequest(name="dp3", processes=2)],
        )

    assert [issue.code for issue in exc.value.issues] == ["resource_processes_oversubscribed"]
    assert exc.value.issues[0].option == "resources"


def test_preflight_reports_slurm_process_oversubscription():
    config = ExecutionConfig(
        task_runner="external_dask",
        dask_scheduler="tcp://scheduler:8786",
        batch_system="slurm",
        max_nodes=1,
    )

    with pytest.raises(PreflightError) as exc:
        preflight_execution(
            config,
            resource_requests=[ResourceRequest(name="dp3", processes=2)],
        )

    assert [issue.code for issue in exc.value.issues] == ["slurm_processes_oversubscribed"]
    assert exc.value.issues[0].option == "resources"
