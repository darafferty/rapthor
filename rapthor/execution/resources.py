"""Resource declarations for external commands."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from rapthor.execution.config import ExecutionConfig


@dataclass(frozen=True)
class ResourceRequest:
    """CPU, memory, process, and MPI requirements for a command task."""

    name: str = "command"
    threads: int = 1
    processes: int = 1
    memory_gb: Optional[int] = None
    use_mpi: bool = False
    exclusive: bool = False

    def __post_init__(self):
        if self.threads < 1:
            raise ValueError("threads must be >= 1")
        if self.processes < 1:
            raise ValueError("processes must be >= 1")
        if self.memory_gb is not None and self.memory_gb < 1:
            raise ValueError("memory_gb must be >= 1 when set")

    @classmethod
    def from_execution_config(
        cls,
        execution_config: ExecutionConfig,
        name: str = "command",
        threads: Optional[int] = None,
        use_mpi: bool = False,
    ) -> "ResourceRequest":
        """Build a conservative request from execution settings."""
        request_threads = threads or execution_config.cpus_per_task or 1
        memory_gb = execution_config.mem_per_node_gb or None
        return cls(
            name=name,
            threads=request_threads,
            memory_gb=memory_gb,
            use_mpi=use_mpi,
            exclusive=use_mpi,
        )


def collect_resource_issues(
    resource_request: ResourceRequest,
    execution_config: ExecutionConfig,
) -> list[tuple[str, str]]:
    """Collect resource validation issues for a command request."""
    issues = []
    if execution_config.cpus_per_task and resource_request.threads > execution_config.cpus_per_task:
        issues.append(
            (
                "resource_threads_oversubscribed",
                f"{resource_request.name} requests {resource_request.threads} threads, "
                f"but cpus_per_task is {execution_config.cpus_per_task}",
            )
        )

    if (
        resource_request.memory_gb
        and execution_config.mem_per_node_gb
        and resource_request.memory_gb > execution_config.mem_per_node_gb
    ):
        issues.append(
            (
                "resource_memory_oversubscribed",
                f"{resource_request.name} requests {resource_request.memory_gb} GB, "
                f"but mem_per_node_gb is {execution_config.mem_per_node_gb}",
            )
        )

    if resource_request.use_mpi and not resource_request.exclusive:
        issues.append(
            (
                "mpi_not_exclusive",
                f"{resource_request.name} uses MPI and must be marked exclusive",
            )
        )

    available_nodes = max(1, execution_config.max_nodes)
    if resource_request.use_mpi and resource_request.processes > available_nodes:
        issues.append(
            (
                "mpi_processes_oversubscribed",
                f"{resource_request.name} requests {resource_request.processes} MPI processes, "
                f"but max_nodes is {execution_config.max_nodes}",
            )
        )

    return issues


def collect_resource_request_issues(
    resource_requests: Iterable[ResourceRequest],
    execution_config: ExecutionConfig,
) -> list[tuple[str, str]]:
    """Collect validation issues across several command requests."""
    issues = []
    for resource_request in resource_requests:
        issues.extend(collect_resource_issues(resource_request, execution_config))
    return issues


def validate_resource_request(
    resource_request: ResourceRequest,
    execution_config: ExecutionConfig,
) -> ResourceRequest:
    """Raise when a command request would oversubscribe the execution config."""
    issues = collect_resource_issues(resource_request, execution_config)
    if issues:
        raise ValueError("; ".join(message for _, message in issues))
    return resource_request


def thread_environment(resource_request: ResourceRequest) -> Mapping[str, str]:
    """Return thread-related environment variables for external tools."""
    threads = str(resource_request.threads)
    return {
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
    }
