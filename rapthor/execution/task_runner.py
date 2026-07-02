"""Prefect task-runner construction helpers."""

from dataclasses import dataclass
from typing import Optional

from rapthor.execution.config import ExecutionConfig


class MissingPrefectDaskError(RuntimeError):
    """Raised when prefect-dask is required but not installed."""


class DaskSchedulerConnectionError(RuntimeError):
    """Raised when an external Dask scheduler cannot be used."""


DASK_SCHEDULER_CHECK_TIMEOUT = "30s"
LOCAL_DASK_WAIT_TIMEOUT = "60s"


@dataclass
class LocalDaskClusterHandle:
    """A local Dask scheduler/client pair managed for one Rapthor run."""

    cluster: object
    client: object
    scheduler_address: str
    dashboard_url: Optional[str]
    worker_count: int

    def close(self) -> None:
        close_client = getattr(self.client, "close", None)
        if close_client is not None:
            close_client()
        close_cluster = getattr(self.cluster, "close", None)
        if close_cluster is not None:
            close_cluster()


def _load_dask_task_runner_cls():
    try:
        from prefect_dask import DaskTaskRunner
    except ImportError as err:
        raise MissingPrefectDaskError(
            "prefect-dask is required for local_dask and external_dask task runners"
        ) from err
    return DaskTaskRunner


def _load_dask_client_cls():
    try:
        from dask.distributed import Client
    except ImportError as err:
        raise MissingPrefectDaskError(
            "dask.distributed is required to validate external Dask schedulers"
        ) from err
    return Client


def _load_local_dask_cluster_classes():
    try:
        from dask.distributed import Client, LocalCluster
    except ImportError as err:
        raise MissingPrefectDaskError(
            "dask.distributed is required to start a local Dask scheduler"
        ) from err
    return LocalCluster, Client


def _load_thread_pool_task_runner_cls():
    from prefect.task_runners import ThreadPoolTaskRunner

    return ThreadPoolTaskRunner


def local_cluster_kwargs(execution_config: ExecutionConfig) -> dict:
    """Build conservative kwargs for a local Dask cluster."""
    kwargs = {
        "n_workers": execution_config.local_dask_worker_count,
        "threads_per_worker": execution_config.local_dask_threads_per_worker,
    }
    if execution_config.mem_per_node_gb:
        kwargs["memory_limit"] = f"{execution_config.mem_per_node_gb}GB"
    if execution_config.dask_dashboard_address:
        kwargs["dashboard_address"] = execution_config.dask_dashboard_address
    return kwargs


def start_local_dask_cluster(
    execution_config: ExecutionConfig,
    *,
    cluster_cls=None,
    client_cls=None,
    wait_timeout: str = LOCAL_DASK_WAIT_TIMEOUT,
) -> LocalDaskClusterHandle:
    """Start one local Dask scheduler for a Rapthor run."""
    if cluster_cls is None or client_cls is None:
        loaded_cluster_cls, loaded_client_cls = _load_local_dask_cluster_classes()
        cluster_cls = cluster_cls or loaded_cluster_cls
        client_cls = client_cls or loaded_client_cls

    worker_count = execution_config.local_dask_worker_count
    cluster = cluster_cls(**local_cluster_kwargs(execution_config))
    client = None
    try:
        client = client_cls(cluster)
        wait_for_workers = getattr(client, "wait_for_workers", None)
        if wait_for_workers is not None:
            wait_for_workers(worker_count, timeout=wait_timeout)
    except Exception:
        if client is not None:
            close_client = getattr(client, "close", None)
            if close_client is not None:
                close_client()
        close_cluster = getattr(cluster, "close", None)
        if close_cluster is not None:
            close_cluster()
        raise

    return LocalDaskClusterHandle(
        cluster=cluster,
        client=client,
        scheduler_address=cluster.scheduler_address,
        dashboard_url=getattr(cluster, "dashboard_link", None),
        worker_count=worker_count,
    )


def check_dask_scheduler(
    address: str,
    client_cls=None,
    timeout: str = DASK_SCHEDULER_CHECK_TIMEOUT,
) -> int:
    """Check that an external Dask scheduler is reachable and has workers."""
    runner_client_cls = client_cls or _load_dask_client_cls()
    client = None
    try:
        client = runner_client_cls(address, timeout=timeout)
        scheduler_info = client.scheduler_info()
    except Exception as err:
        raise DaskSchedulerConnectionError(
            f"Could not connect to Dask scheduler at {address!r}: {err}"
        ) from err
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                close()

    workers = scheduler_info.get("workers", {})
    if not workers:
        raise DaskSchedulerConnectionError(
            f"Dask scheduler at {address!r} has no connected workers"
        )
    return len(workers)


def build_task_runner(
    execution_config: ExecutionConfig,
    dask_task_runner_cls=None,
    thread_pool_task_runner_cls=None,
    dask_client_cls=None,
):
    """Build the configured Prefect task runner.

    The `sync` runner uses Prefect's thread-pool runner with one worker so tests
    keep deterministic task ordering without importing prefect-dask.
    """
    if execution_config.task_runner == "sync":
        runner_cls = thread_pool_task_runner_cls or _load_thread_pool_task_runner_cls()
        return runner_cls(max_workers=1)

    runner_cls = dask_task_runner_cls or _load_dask_task_runner_cls()
    if execution_config.task_runner == "external_dask":
        scheduler = execution_config.resolved_dask_scheduler()
        if not scheduler:
            raise ValueError("external_dask requires a dask_scheduler value")
        check_dask_scheduler(scheduler, client_cls=dask_client_cls)
        return runner_cls(address=scheduler)
    return runner_cls(
        cluster_class="dask.distributed.LocalCluster",
        cluster_kwargs=local_cluster_kwargs(execution_config),
    )


def run_flow_with_task_runner(
    prefect_flow,
    *flow_args,
    execution_config: Optional[ExecutionConfig] = None,
    flow_run_name: Optional[str] = None,
    **flow_kwargs,
):
    """Run a Prefect flow with the task runner requested by execution config."""
    config = execution_config or ExecutionConfig()
    task_runner = build_task_runner(config)
    flow_options = {"task_runner": task_runner}
    if flow_run_name is not None:
        flow_options["flow_run_name"] = flow_run_name
    configured_flow = prefect_flow.with_options(**flow_options)
    return configured_flow(*flow_args, execution_config=config, **flow_kwargs)
