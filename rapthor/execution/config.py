"""Configuration objects for Rapthor's Python execution layer."""

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

TASK_RUNNERS = ("local_dask", "external_dask", "sync")
COMMAND_PROFILE_MODES = ("auto", "time", "perf", "off")
PREFECT_API_MODES = ("auto", "external", "ephemeral")
DASK_SCHEDULER_ENV = "DASK_SCHEDULER"
PREFECT_API_URL_ENV = "PREFECT_API_URL"


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


def _as_positive_int(value: Any, name: str) -> int:
    parsed = _as_non_negative_int(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def _as_choice(value: Any, name: str, choices: tuple[str, ...]) -> str:
    if value is None:
        return choices[0]
    parsed = str(value).strip().lower()
    if parsed not in choices:
        allowed = ", ".join(map(repr, choices))
        raise ValueError(f"{name} must be one of {allowed}")
    return parsed


def _cluster_settings(parset: Mapping[str, Any]) -> Mapping[str, Any]:
    return parset.get("cluster_specific", parset.get("cluster", {}))


def dask_scheduler_from_environment(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return the scheduler address exported by Slurm launch scripts, if any."""
    environment = os.environ if environ is None else environ
    return _optional_str(environment.get(DASK_SCHEDULER_ENV))


def prefect_api_url_from_environment(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return the Prefect API URL exported for this process, if any."""
    environment = os.environ if environ is None else environ
    return _optional_str(environment.get(PREFECT_API_URL_ENV))


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime settings for the Prefect/Dask execution path."""

    task_runner: str = "local_dask"
    prefect_api_mode: str = "auto"
    prefect_api_url: Optional[str] = None
    dask_scheduler: Optional[str] = None
    dask_dashboard_address: Optional[str] = None
    stream_output: bool = True
    retries: int = 0
    log_commands: bool = True
    command_profile: str = "auto"
    publish_fits_previews: bool = False
    publish_postage_stamp_previews: bool = False
    postage_stamp_preview_count: int = 5
    postage_stamp_preview_size_px: int = 96
    batch_system: str = "single_machine"
    max_nodes: int = 1
    local_dask_workers: int = 0
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

        Rapthor does not expose a public backend selector. These settings
        describe how the Prefect/Dask execution path should run.
        """
        cluster = _cluster_settings(parset)
        scheduler = (
            _optional_str(cluster.get("dask_scheduler")) or dask_scheduler_from_environment()
        )
        prefect_api_mode = _as_choice(
            cluster.get("prefect_api_mode", "auto"),
            "prefect_api_mode",
            PREFECT_API_MODES,
        )
        prefect_api_url = (
            _optional_str(cluster.get("prefect_api_url")) or prefect_api_url_from_environment()
        )
        task_runner = cluster.get("prefect_task_runner")
        if task_runner is None:
            task_runner = "external_dask" if scheduler else "local_dask"
        task_runner = str(task_runner)
        if task_runner not in TASK_RUNNERS:
            allowed = ", ".join(map(repr, TASK_RUNNERS))
            raise ValueError(f"prefect_task_runner must be one of {allowed}")

        return cls(
            task_runner=task_runner,
            prefect_api_mode=prefect_api_mode,
            prefect_api_url=prefect_api_url,
            dask_scheduler=scheduler,
            dask_dashboard_address=_optional_str(cluster.get("dask_dashboard_address")),
            stream_output=_as_bool(
                cluster.get("prefect_stream_output", True), "prefect_stream_output"
            ),
            retries=_as_non_negative_int(cluster.get("prefect_retries", 0), "prefect_retries"),
            log_commands=_as_bool(
                cluster.get("prefect_log_commands", True), "prefect_log_commands"
            ),
            command_profile=_as_choice(
                cluster.get("prefect_command_profile", "auto"),
                "prefect_command_profile",
                COMMAND_PROFILE_MODES,
            ),
            publish_fits_previews=_as_bool(
                cluster.get("prefect_publish_fits_previews", False),
                "prefect_publish_fits_previews",
            ),
            publish_postage_stamp_previews=_as_bool(
                cluster.get("prefect_publish_postage_stamp_previews", False),
                "prefect_publish_postage_stamp_previews",
            ),
            postage_stamp_preview_count=_as_non_negative_int(
                cluster.get("prefect_postage_stamp_preview_count", 5),
                "prefect_postage_stamp_preview_count",
            ),
            postage_stamp_preview_size_px=_as_positive_int(
                cluster.get("prefect_postage_stamp_preview_size_px", 96),
                "prefect_postage_stamp_preview_size_px",
            ),
            batch_system=str(cluster.get("batch_system", "single_machine")),
            max_nodes=_as_non_negative_int(cluster.get("max_nodes", 1), "max_nodes"),
            local_dask_workers=_as_non_negative_int(
                cluster.get("local_dask_workers", 0), "local_dask_workers"
            ),
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

    def resolved_dask_scheduler(self, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
        """Return the configured Dask scheduler, including environment fallback."""
        return self.dask_scheduler or dask_scheduler_from_environment(environ)

    @property
    def local_dask_worker_count(self) -> int:
        """Return the worker count used for a local Dask cluster."""
        if self.local_dask_workers:
            return self.local_dask_workers
        return max(1, self.max_nodes)

    @property
    def local_dask_threads_per_worker(self) -> int:
        """Return the thread count used for each local Dask worker."""
        return max(1, self.cpus_per_task or 1)
