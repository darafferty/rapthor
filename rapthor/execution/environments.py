"""Explicit environment policies for external command tasks.

String values set variables in the child process; None removes inherited
variables. Helpers return fresh mappings and never read or change os.environ.
"""

from typing import Mapping, Optional

from rapthor.execution.resources import ResourceRequest

EnvironmentOverrides = Mapping[str, Optional[str]]


def dp3_environment() -> EnvironmentOverrides:
    """Return overrides for DP3 calibration and prediction subprocesses."""
    # Dask's nanny sets this for Python workers. In glibc it also disables
    # adaptive mmap thresholds, which can penalize FastPredict's temporary
    # allocations. Restore the allocator default only for this DP3 process.
    return {"MALLOC_TRIM_THRESHOLD_": None}


def thread_environment(resource_request: ResourceRequest) -> EnvironmentOverrides:
    """Return common thread variables using the command's requested thread count."""
    threads = str(resource_request.threads)
    return {
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
    }


def wsclean_environment(resource_request: ResourceRequest) -> EnvironmentOverrides:
    """Return overrides for WSClean imaging using the command's resources."""
    # Restore glibc's adaptive allocation thresholds for WSClean's buffers.
    environment: dict[str, Optional[str]] = {"MALLOC_TRIM_THRESHOLD_": None}
    if not resource_request.use_mpi:
        environment["DUCC0_NUM_THREADS"] = str(resource_request.threads)
        return environment
    environment.update(thread_environment(resource_request))
    # WSClean rejects multi-threaded OpenBLAS because it interferes with
    # WSClean's own thread pool. Keep the requested OMP/WSClean thread count,
    # but export a single OpenBLAS thread to every MPI rank.
    environment["OPENBLAS_NUM_THREADS"] = "1"
    return environment
