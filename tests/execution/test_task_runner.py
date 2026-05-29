import pytest

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.task_runner import (
    MissingPrefectDaskError,
    build_task_runner,
    local_cluster_kwargs,
)


class FakeDaskTaskRunner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_sync_task_runner_does_not_require_prefect_dask():
    assert build_task_runner(ExecutionConfig(task_runner="sync")) is None


def test_local_cluster_kwargs_are_conservative():
    kwargs = local_cluster_kwargs(ExecutionConfig(max_nodes=2, cpus_per_task=4))

    assert kwargs == {"n_workers": 2, "threads_per_worker": 4}


def test_build_local_dask_task_runner_with_injected_class():
    runner = build_task_runner(
        ExecutionConfig(task_runner="local_dask", max_nodes=2, cpus_per_task=4),
        dask_task_runner_cls=FakeDaskTaskRunner,
    )

    assert runner.kwargs == {
        "cluster_class": "dask.distributed.LocalCluster",
        "cluster_kwargs": {"n_workers": 2, "threads_per_worker": 4},
    }


def test_build_external_dask_task_runner_with_injected_class():
    runner = build_task_runner(
        ExecutionConfig(task_runner="external_dask", dask_scheduler="tcp://scheduler:8786"),
        dask_task_runner_cls=FakeDaskTaskRunner,
    )

    assert runner.kwargs == {"address": "tcp://scheduler:8786"}


def test_build_external_dask_task_runner_requires_scheduler():
    with pytest.raises(ValueError, match="dask_scheduler"):
        build_task_runner(
            ExecutionConfig(task_runner="external_dask"),
            dask_task_runner_cls=FakeDaskTaskRunner,
        )


def test_build_task_runner_requires_prefect_dask_without_injection(monkeypatch):
    def missing_dask_task_runner():
        raise MissingPrefectDaskError("prefect-dask is required")

    monkeypatch.setattr(
        "rapthor.execution.task_runner._load_dask_task_runner_cls",
        missing_dask_task_runner,
    )

    with pytest.raises(MissingPrefectDaskError, match="prefect-dask"):
        build_task_runner(ExecutionConfig(task_runner="local_dask"))
