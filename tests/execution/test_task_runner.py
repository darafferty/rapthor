import pytest

from rapthor.execution.config import DASK_SCHEDULER_ENV, ExecutionConfig
from rapthor.execution.task_runner import (
    DaskSchedulerConnectionError,
    MissingPrefectDaskError,
    build_task_runner,
    check_dask_scheduler,
    local_cluster_kwargs,
    run_flow_with_task_runner,
    start_local_dask_cluster,
)


class FakeDaskTaskRunner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeThreadPoolTaskRunner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeDaskClient:
    calls = []

    def __init__(self, address, timeout):
        self.calls.append((address, timeout))

    def scheduler_info(self):
        return {"workers": {"worker-1": {}}}

    def close(self):
        self.closed = True


class FakeLocalCluster:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scheduler_address = "tcp://127.0.0.1:8786"
        self.dashboard_link = "http://127.0.0.1:8787/status"
        self.closed = False
        self.instances.append(self)

    def close(self):
        self.closed = True


class FakeLocalClient:
    instances = []

    def __init__(self, cluster):
        self.cluster = cluster
        self.waits = []
        self.closed = False
        self.instances.append(self)

    def wait_for_workers(self, worker_count, timeout):
        self.waits.append((worker_count, timeout))

    def close(self):
        self.closed = True


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


class FakeTagContext:
    def __init__(self, calls, tags):
        self.calls = calls
        self.tags = tags

    def __enter__(self):
        self.calls.append(("enter", self.tags))

    def __exit__(self, exc_type, exc_value, traceback):
        self.calls.append(("exit", self.tags))


def test_sync_task_runner_uses_single_worker_thread_pool():
    runner = build_task_runner(
        ExecutionConfig(task_runner="sync"),
        thread_pool_task_runner_cls=FakeThreadPoolTaskRunner,
    )

    assert runner.kwargs == {"max_workers": 1}


def test_local_cluster_kwargs_are_conservative():
    kwargs = local_cluster_kwargs(ExecutionConfig(local_dask_workers=2, cpus_per_task=4))

    assert kwargs == {"n_workers": 2, "threads_per_worker": 1, "processes": True}


def test_local_cluster_kwargs_default_to_single_worker_capacity():
    kwargs = local_cluster_kwargs(ExecutionConfig(max_nodes=0, cpus_per_task=0))

    assert kwargs == {"n_workers": 1, "threads_per_worker": 1, "processes": True}


def test_local_cluster_kwargs_include_memory_limit():
    kwargs = local_cluster_kwargs(
        ExecutionConfig(local_dask_workers=2, cpus_per_task=4, mem_per_node_gb=128)
    )

    assert kwargs == {
        "n_workers": 2,
        "threads_per_worker": 1,
        "processes": True,
        "memory_limit": "128GB",
    }


def test_local_cluster_kwargs_include_dashboard_address():
    kwargs = local_cluster_kwargs(
        ExecutionConfig(local_dask_workers=2, cpus_per_task=4, dask_dashboard_address=":8787")
    )

    assert kwargs == {
        "n_workers": 2,
        "threads_per_worker": 1,
        "processes": True,
        "dashboard_address": ":8787",
    }


def test_start_local_dask_cluster_returns_managed_handle():
    FakeLocalCluster.instances = []
    FakeLocalClient.instances = []

    handle = start_local_dask_cluster(
        ExecutionConfig(local_dask_workers=2, cpus_per_task=4, dask_dashboard_address=":8787"),
        cluster_cls=FakeLocalCluster,
        client_cls=FakeLocalClient,
        wait_timeout="5s",
    )

    assert handle.scheduler_address == "tcp://127.0.0.1:8786"
    assert handle.dashboard_url == "http://127.0.0.1:8787/status"
    assert handle.worker_count == 2
    assert FakeLocalCluster.instances[0].kwargs == {
        "n_workers": 2,
        "threads_per_worker": 1,
        "processes": True,
        "dashboard_address": ":8787",
    }
    assert FakeLocalClient.instances[0].waits == [(2, "5s")]

    handle.close()

    assert FakeLocalClient.instances[0].closed is True
    assert FakeLocalCluster.instances[0].closed is True


def test_start_local_dask_cluster_cleans_up_when_workers_do_not_start():
    FakeLocalCluster.instances = []

    class FailingLocalClient(FakeLocalClient):
        def wait_for_workers(self, worker_count, timeout):
            raise TimeoutError("no workers")

    with pytest.raises(TimeoutError, match="no workers"):
        start_local_dask_cluster(
            ExecutionConfig(local_dask_workers=2),
            cluster_cls=FakeLocalCluster,
            client_cls=FailingLocalClient,
        )

    assert FakeLocalCluster.instances[0].closed is True


def test_build_local_dask_task_runner_with_injected_class():
    runner = build_task_runner(
        ExecutionConfig(task_runner="local_dask", local_dask_workers=2, cpus_per_task=4),
        dask_task_runner_cls=FakeDaskTaskRunner,
    )

    assert runner.kwargs == {
        "cluster_class": "dask.distributed.LocalCluster",
        "cluster_kwargs": {"n_workers": 2, "threads_per_worker": 1, "processes": True},
    }


def test_build_external_dask_task_runner_with_injected_class():
    FakeDaskClient.calls = []

    runner = build_task_runner(
        ExecutionConfig(task_runner="external_dask", dask_scheduler="tcp://scheduler:8786"),
        dask_task_runner_cls=FakeDaskTaskRunner,
        dask_client_cls=FakeDaskClient,
    )

    assert runner.kwargs == {"address": "tcp://scheduler:8786"}
    assert FakeDaskClient.calls == [("tcp://scheduler:8786", "30s")]


def test_build_external_dask_task_runner_uses_environment_scheduler(monkeypatch):
    monkeypatch.setenv(DASK_SCHEDULER_ENV, "tcp://env-scheduler:8786")
    FakeDaskClient.calls = []

    runner = build_task_runner(
        ExecutionConfig(task_runner="external_dask"),
        dask_task_runner_cls=FakeDaskTaskRunner,
        dask_client_cls=FakeDaskClient,
    )

    assert runner.kwargs == {"address": "tcp://env-scheduler:8786"}
    assert FakeDaskClient.calls == [("tcp://env-scheduler:8786", "30s")]


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


def test_run_flow_with_task_runner_applies_flow_run_name(monkeypatch):
    runner = object()
    config = ExecutionConfig(task_runner="sync")
    flow = FakeFlow()

    monkeypatch.setattr(
        "rapthor.execution.task_runner.build_task_runner",
        lambda execution_config: runner,
    )

    run_flow_with_task_runner(
        flow,
        "payload",
        execution_config=config,
        flow_run_name="calibrate_dd_2",
    )

    assert flow.options == {"task_runner": runner, "flow_run_name": "calibrate_dd_2"}


def test_run_flow_with_task_runner_applies_prefect_run_tags(monkeypatch):
    runner = object()
    config = ExecutionConfig(task_runner="sync", run_tags=("demo", "multi-sector"))
    flow = FakeFlow()
    tag_calls = []

    monkeypatch.setattr(
        "rapthor.execution.task_runner.build_task_runner",
        lambda execution_config: runner,
    )
    monkeypatch.setattr(
        "rapthor.execution.task_runner.prefect_tags",
        lambda *tags: FakeTagContext(tag_calls, tags),
    )

    run_flow_with_task_runner(flow, "payload", execution_config=config)

    assert tag_calls == [
        ("enter", ("demo", "multi-sector")),
        ("exit", ("demo", "multi-sector")),
    ]


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
        closed = False

        def __init__(self, address, timeout):
            self.address = address

        def scheduler_info(self):
            return {"workers": {}}

        def close(self):
            FakeClient.closed = True

    with pytest.raises(DaskSchedulerConnectionError, match="no connected workers"):
        check_dask_scheduler("tcp://scheduler:8786", client_cls=FakeClient)

    assert FakeClient.closed is True


def test_check_dask_scheduler_wraps_connection_errors():
    class FailingClient:
        def __init__(self, address, timeout):
            raise OSError("connection refused")

    with pytest.raises(DaskSchedulerConnectionError, match="connection refused"):
        check_dask_scheduler("tcp://scheduler:8786", client_cls=FailingClient)
