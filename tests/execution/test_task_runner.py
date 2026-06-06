import pytest

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.runtime import run_flow_with_task_runner
from rapthor.execution.task_runner import (
    MissingPrefectDaskError,
    build_task_runner,
    local_cluster_kwargs,
)


class FakeDaskTaskRunner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeThreadPoolTaskRunner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeConfiguredFlow:
    def __init__(self, parent):
        self.parent = parent
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"args": args, "kwargs": kwargs}


class FakeFlow:
    def __init__(self):
        self.options = None
        self.configured_flow = FakeConfiguredFlow(self)

    def with_options(self, **kwargs):
        self.options = kwargs
        return self.configured_flow


def test_sync_task_runner_uses_single_worker_thread_pool():
    runner = build_task_runner(
        ExecutionConfig(task_runner="sync"),
        thread_pool_task_runner_cls=FakeThreadPoolTaskRunner,
    )

    assert runner.kwargs == {"max_workers": 1}


def test_local_cluster_kwargs_are_conservative():
    kwargs = local_cluster_kwargs(ExecutionConfig(max_nodes=2, cpus_per_task=4))

    assert kwargs == {"n_workers": 2, "threads_per_worker": 4}


def test_local_cluster_kwargs_default_to_single_worker_capacity():
    kwargs = local_cluster_kwargs(ExecutionConfig(max_nodes=0, cpus_per_task=0))

    assert kwargs == {"n_workers": 1, "threads_per_worker": 1}


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


def test_run_flow_with_task_runner_applies_configured_runner(monkeypatch):
    runner = object()
    config = ExecutionConfig(task_runner="external_dask", dask_scheduler="tcp://scheduler:8786")
    flow = FakeFlow()

    monkeypatch.setattr(
        "rapthor.execution.flows.runtime.build_task_runner",
        lambda execution_config: runner,
    )

    result = run_flow_with_task_runner(flow, "payload", execution_config=config)

    assert flow.options == {"task_runner": runner}
    assert result == {
        "args": ("payload",),
        "kwargs": {"execution_config": config},
    }
