"""Configuration objects for Rapthor's Python execution layer."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

TASK_RUNNERS = ("local_dask", "external_dask", "sync")


def _optional_str(value: Any) -> Optional[str]:
    if value in (None, "", "None"):
        return None
    return str(value)


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value")


def _as_non_negative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} must be an integer") from err
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


def _cluster_settings(parset: Mapping[str, Any]) -> Mapping[str, Any]:
    return parset.get("cluster_specific", parset.get("cluster", {}))


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime settings for the Prefect/Dask execution path."""

    task_runner: str = "local_dask"
    dask_scheduler: Optional[str] = None
    stream_output: bool = True
    retries: int = 0
    log_commands: bool = True
    batch_system: str = "single_machine"
    max_nodes: int = 1
    cpus_per_task: int = 0
    mem_per_node_gb: int = 0
    use_container: bool = False
    container_type: Optional[str] = None
    local_scratch_dir: Optional[str] = None
    global_scratch_dir: Optional[str] = None
    deprecated_dir_local: Optional[str] = None

    @classmethod
    def from_parset(cls, parset: Mapping[str, Any]) -> "ExecutionConfig":
        """Build execution settings from a Rapthor parset dictionary.

        The migration branch does not expose a public backend selector. These
        settings describe how the single Python execution path should run.
        """
        cluster = _cluster_settings(parset)
        scheduler = _optional_str(cluster.get("dask_scheduler"))
        task_runner = cluster.get("prefect_task_runner")
        if task_runner is None:
            task_runner = "external_dask" if scheduler else "local_dask"
        task_runner = str(task_runner)
        if task_runner not in TASK_RUNNERS:
            allowed = ", ".join(map(repr, TASK_RUNNERS))
            raise ValueError(f"prefect_task_runner must be one of {allowed}")

        return cls(
            task_runner=task_runner,
            dask_scheduler=scheduler,
            stream_output=_as_bool(
                cluster.get("prefect_stream_output", True), "prefect_stream_output"
            ),
            retries=_as_non_negative_int(cluster.get("prefect_retries", 0), "prefect_retries"),
            log_commands=_as_bool(
                cluster.get("prefect_log_commands", True), "prefect_log_commands"
            ),
            batch_system=str(cluster.get("batch_system", "single_machine")),
            max_nodes=_as_non_negative_int(cluster.get("max_nodes", 1), "max_nodes"),
            cpus_per_task=_as_non_negative_int(cluster.get("cpus_per_task", 0), "cpus_per_task"),
            mem_per_node_gb=_as_non_negative_int(
                cluster.get("mem_per_node_gb", 0), "mem_per_node_gb"
            ),
            use_container=_as_bool(cluster.get("use_container", False), "use_container"),
            container_type=_optional_str(cluster.get("container_type")),
            local_scratch_dir=_optional_str(cluster.get("local_scratch_dir")),
            global_scratch_dir=_optional_str(cluster.get("global_scratch_dir")),
            deprecated_dir_local=_optional_str(cluster.get("dir_local")),
        )

    @property
    def effective_local_scratch_dir(self) -> Optional[str]:
        """Return the preferred local scratch directory, including legacy fallback."""
        return self.local_scratch_dir or self.deprecated_dir_local
