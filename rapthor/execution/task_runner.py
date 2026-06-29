"""Prefect task-runner construction helpers."""

from typing import Optional

from rapthor.execution.config import ExecutionConfig


class MissingPrefectDaskError(RuntimeError):
    """Raised when prefect-dask is required but not installed."""


class DaskSchedulerConnectionError(RuntimeError):
    """Raised when an external Dask scheduler cannot be used."""


DASK_SCHEDULER_CHECK_TIMEOUT = "30s"


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
    **flow_kwargs,
):
    """Run a Prefect flow with the task runner requested by execution config."""
    config = execution_config or ExecutionConfig()
    task_runner = build_task_runner(config)
    configured_flow = prefect_flow.with_options(task_runner=task_runner)
    return configured_flow(*flow_args, execution_config=config, **flow_kwargs)
