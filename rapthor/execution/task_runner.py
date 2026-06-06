"""Prefect task-runner construction helpers."""

from rapthor.execution.config import ExecutionConfig


class MissingPrefectDaskError(RuntimeError):
    """Raised when prefect-dask is required but not installed."""


def _load_dask_task_runner_cls():
    try:
        from prefect_dask import DaskTaskRunner
    except ImportError as err:
        raise MissingPrefectDaskError(
            "prefect-dask is required for local_dask and external_dask task runners"
        ) from err
    return DaskTaskRunner


def _load_thread_pool_task_runner_cls():
    from prefect.task_runners import ThreadPoolTaskRunner

    return ThreadPoolTaskRunner


def local_cluster_kwargs(execution_config: ExecutionConfig) -> dict:
    """Build conservative kwargs for a local Dask cluster."""
    return {
        "n_workers": execution_config.local_dask_worker_count,
        "threads_per_worker": execution_config.local_dask_threads_per_worker,
    }


def build_task_runner(
    execution_config: ExecutionConfig,
    dask_task_runner_cls=None,
    thread_pool_task_runner_cls=None,
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
        if not execution_config.dask_scheduler:
            raise ValueError("external_dask requires a dask_scheduler value")
        return runner_cls(address=execution_config.dask_scheduler)
    return runner_cls(
        cluster_class="dask.distributed.LocalCluster",
        cluster_kwargs=local_cluster_kwargs(execution_config),
    )
