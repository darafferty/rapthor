import pytest

from rapthor.execution.config import DASK_SCHEDULER_ENV, ExecutionConfig
from rapthor.execution.task_runner import (
    DaskSchedulerConnectionError,
    MissingPrefectDaskError,
    build_task_runner,
    check_dask_scheduler,
    local_cluster_kwargs,
    run_flow_with_task_runner,
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


def test_local_cluster_kwargs_include_memory_limit():
    kwargs = local_cluster_kwargs(
        ExecutionConfig(max_nodes=2, cpus_per_task=4, mem_per_node_gb=128)
    )

    assert kwargs == {"n_workers": 2, "threads_per_worker": 4, "memory_limit": "128GB"}


def test_local_cluster_kwargs_include_dashboard_address():
    kwargs = local_cluster_kwargs(
        ExecutionConfig(max_nodes=2, cpus_per_task=4, dask_dashboard_address=":8787")
    )

    assert kwargs == {"n_workers": 2, "threads_per_worker": 4, "dashboard_address": ":8787"}


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


def test_build_external_dask_task_runner_uses_environment_scheduler(monkeypatch):
    monkeypatch.setenv(DASK_SCHEDULER_ENV, "tcp://env-scheduler:8786")

    runner = build_task_runner(
        ExecutionConfig(task_runner="external_dask"),
        dask_task_runner_cls=FakeDaskTaskRunner,
    )

    assert runner.kwargs == {"address": "tcp://env-scheduler:8786"}


def test_build_external_dask_task_runner_requires_scheduler(monkeypatch):
    monkeypatch.delenv(DASK_SCHEDULER_ENV, raising=False)

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
        "rapthor.execution.task_runner.build_task_runner",
        lambda execution_config: runner,
    )

    result = run_flow_with_task_runner(flow, "payload", execution_config=config)

    assert flow.options == {"task_runner": runner}
    assert result == {
        "args": ("payload",),
        "kwargs": {"execution_config": config},
    }


def test_check_dask_scheduler_returns_worker_count():
    class FakeClient:
        calls = []

        def __init__(self, address, timeout):
            self.calls.append((address, timeout))

        def scheduler_info(self):
            return {"workers": {"worker-1": {}, "worker-2": {}}}

        def close(self):
            self.closed = True

    worker_count = check_dask_scheduler(
        "tcp://scheduler:8786",
        client_cls=FakeClient,
        timeout="1s",
    )

    assert worker_count == 2
    assert FakeClient.calls == [("tcp://scheduler:8786", "1s")]


def test_check_dask_scheduler_rejects_scheduler_without_workers():
    class FakeClient:
        def __init__(self, address, timeout):
            self.address = address

        def scheduler_info(self):
            return {"workers": {}}

        def close(self):
            pass

    with pytest.raises(DaskSchedulerConnectionError, match="no connected workers"):
        check_dask_scheduler("tcp://scheduler:8786", client_cls=FakeClient)


def test_check_dask_scheduler_wraps_connection_errors():
    class FailingClient:
        def __init__(self, address, timeout):
            raise OSError("connection refused")

    with pytest.raises(DaskSchedulerConnectionError, match="connection refused"):
        check_dask_scheduler("tcp://scheduler:8786", client_cls=FailingClient)
