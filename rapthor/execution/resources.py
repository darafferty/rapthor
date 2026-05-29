"""Resource declarations for external commands."""

from dataclasses import dataclass
from typing import Mapping, Optional

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


def thread_environment(resource_request: ResourceRequest) -> Mapping[str, str]:
    """Return thread-related environment variables for external tools."""
    threads = str(resource_request.threads)
    return {
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
    }
