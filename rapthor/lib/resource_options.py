"""Resolve user CPU settings before constructing execution payloads.

Local workers have fixed, equal shares of the CPUs available to Rapthor.
External workers each own one host allocation. Zero means automatic; positive
thread settings must fit their worker's allocation.
"""

import os
from typing import Mapping, Optional


def available_cpu_count() -> int:
    """Count CPUs allowed by affinity, falling back on the system CPU count."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"The option '{name}' must be a non-negative integer")
    return value


def resolve_tool_threads(options: Mapping, tool: str) -> int:
    """Resolve legacy/raw input dictionaries at a payload boundary."""
    name = f"{tool}_max_threads"
    return nonnegative_integer(options.get(name, 0), name) or int(options["max_threads"])


def resolve_cluster_resources(
    options: Mapping,
    *,
    available_cpus: Optional[int] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> dict:
    """Return resolved settings without changing the caller's dictionary."""
    result = dict(options)
    environment = os.environ if environ is None else environ
    available = available_cpu_count() if available_cpus is None else available_cpus
    slurm = str(options.get("batch_system", "single_machine")).startswith("slurm")
    runner = options.get("prefect_task_runner") or (
        "external_dask"
        if options.get("dask_scheduler") or environment.get("DASK_SCHEDULER")
        else "local_dask"
    )
    requested_nodes = nonnegative_integer(options.get("max_nodes", 0), "max_nodes")
    allocated_nodes = (
        int(environment.get("SLURM_NNODES") or environment.get("SLURM_JOB_NUM_NODES") or 0)
        if slurm
        else 0
    )
    if allocated_nodes < 0:
        raise ValueError("Slurm node count must be positive")
    if allocated_nodes and requested_nodes > allocated_nodes:
        raise ValueError("max_nodes exceeds the Slurm node allocation")
    nodes = requested_nodes or allocated_nodes or (12 if slurm else 1)
    workers = nonnegative_integer(options.get("local_dask_workers", 0), "local_dask_workers")
    # Preserve the local layout when bootstrap connects to its own scheduler.
    slots = options.get("workers_per_node") or (
        (workers or nodes) if runner == "local_dask" and not slurm else 1
    )
    slots = nonnegative_integer(slots, "workers_per_node")
    if not slots:
        raise ValueError("workers_per_node must be positive")
    allocation = available
    if slurm and environment.get("SLURM_CPUS_PER_TASK"):
        allocation = int(environment["SLURM_CPUS_PER_TASK"])
        if allocation < 1:
            raise ValueError("SLURM_CPUS_PER_TASK must be positive")
    budget = nonnegative_integer(options.get("cpus_per_task", 0), "cpus_per_task")
    local = runner != "external_dask" and not slurm
    if local:
        share = available // slots
        if share < 1:
            raise ValueError(f"{slots} local workers exceed the {available} available CPUs")
        if budget > share:
            raise ValueError(
                f"cpus_per_task={budget} with {slots} workers exceeds the "
                f"{available} available CPUs; use at most {share} CPUs per task"
            )
        budget = budget or share
    else:
        if slurm and environment.get("SLURM_CPUS_PER_TASK") and budget > allocation:
            raise ValueError("cpus_per_task exceeds SLURM_CPUS_PER_TASK")
        budget = budget or allocation
    result.update(max_nodes=nodes, workers_per_node=slots, cpus_per_task=budget)
    for name in ("max_threads", "dp3_max_threads", "wsclean_max_threads"):
        requested = nonnegative_integer(options.get(name, 0), name)
        if requested > budget:
            raise ValueError(f"{name}={requested} exceeds cpus_per_task={budget}")
        result[name] = requested or (budget if name == "max_threads" else result["max_threads"])
    filtering = nonnegative_integer(
        options.get("filter_skymodel_ncores", 0), "filter_skymodel_ncores"
    )
    if filtering > budget:
        raise ValueError(f"filter_skymodel_ncores={filtering} exceeds cpus_per_task={budget}")
    result["filter_skymodel_ncores"] = filtering or min(15, result["max_threads"])
    wsclean = result["wsclean_max_threads"]
    deconvolution = nonnegative_integer(
        options.get("deconvolution_threads", 0), "deconvolution_threads"
    )
    if deconvolution > wsclean:
        raise ValueError("deconvolution_threads exceeds the resolved WSClean thread count")
    result["deconvolution_threads"] = deconvolution or max(1, min(14, wsclean * 2 // 5))
    result["parallel_gridding_tasks"] = nonnegative_integer(
        options.get("parallel_gridding_tasks", 0), "parallel_gridding_tasks"
    ) or max(1, wsclean // 8)
    result["max_cores"] = nonnegative_integer(options.get("max_cores", 0), "max_cores") or budget
    return result
