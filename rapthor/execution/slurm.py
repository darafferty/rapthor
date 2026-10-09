"""Helpers for Rapthor's supported Slurm and external-Dask execution mode."""

from dataclasses import dataclass
from typing import Mapping, Optional

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.resources import SLURM_BATCH_SYSTEMS

SLURM_NODE_COUNT_ENV = ("SLURM_NNODES", "SLURM_JOB_NUM_NODES")
SLURM_TASK_COUNT_ENV = ("SLURM_NTASKS",)
SLURM_CPUS_PER_TASK_ENV = "SLURM_CPUS_PER_TASK"


@dataclass(frozen=True)
class SlurmClusterSpec:
    """Resolved Slurm allocation shape used by the external-Dask runner."""

    node_count: int
    task_count: int
    cpus_per_task: int
    worker_count: int
    threads_per_worker: int
    command_threads_per_task: int
    memory_per_node_gb: int


def slurm_cluster_spec(
    execution_config: ExecutionConfig,
    environ: Optional[Mapping[str, str]] = None,
) -> SlurmClusterSpec:
    """Resolve Slurm allocation settings into the Dask worker layout.

    The supported model starts one single-threaded Dask worker per allocated
    node. External command parallelism is tracked separately through the
    ``command_threads_per_task`` budget.
    """
    environment = {} if environ is None else environ
    node_count = (
        _first_positive_int(
            environment,
            SLURM_NODE_COUNT_ENV,
        )
        or execution_config.max_nodes
    )
    if not node_count:
        node_count = 1

    task_count = _first_positive_int(environment, SLURM_TASK_COUNT_ENV) or node_count
    cpus_per_task = (
        execution_config.cpus_per_task
        or _positive_int(environment.get(SLURM_CPUS_PER_TASK_ENV), SLURM_CPUS_PER_TASK_ENV)
        or 1
    )
    return SlurmClusterSpec(
        node_count=node_count,
        task_count=task_count,
        cpus_per_task=cpus_per_task,
        worker_count=node_count,
        threads_per_worker=1,
        command_threads_per_task=cpus_per_task,
        memory_per_node_gb=execution_config.mem_per_node_gb,
    )


def collect_slurm_config_issues(
    execution_config: ExecutionConfig,
    environ: Optional[Mapping[str, str]] = None,
) -> list[tuple[str, str]]:
    """Collect Slurm allocation issues that are independent of command requests."""
    if str(execution_config.batch_system).lower() not in SLURM_BATCH_SYSTEMS:
        return []

    issues = []
    if execution_config.task_runner != "external_dask":
        issues.append(
            (
                "slurm_requires_external_dask",
                "Slurm execution requires prefect_task_runner='external_dask'",
            )
        )

    try:
        spec = slurm_cluster_spec(execution_config, environ=environ)
    except ValueError as err:
        issues.append(("invalid_slurm_allocation", str(err)))
        return issues

    if spec.task_count < spec.node_count:
        issues.append(
            (
                "slurm_tasks_less_than_nodes",
                f"Slurm allocation exposes {spec.task_count} tasks for "
                f"{spec.node_count} nodes; Rapthor expects at least one task per node",
            )
        )

    return issues


def _positive_int(value: Optional[str], name: str) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except ValueError as err:
        raise ValueError(f"{name} must be an integer") from err
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1")
    return parsed


def _first_positive_int(
    environ: Mapping[str, str],
    names: tuple[str, ...],
) -> Optional[int]:
    for name in names:
        parsed = _positive_int(environ.get(name), name)
        if parsed is not None:
            return parsed
    return None
