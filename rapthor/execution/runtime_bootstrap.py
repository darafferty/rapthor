"""Runtime bootstrap checks for the CLI Prefect/Dask entry point."""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Callable, Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

from rapthor.execution.config import PREFECT_API_URL_ENV, ExecutionConfig
from rapthor.execution.task_runner import check_dask_scheduler

log = logging.getLogger("rapthor:runtime")

PREFECT_API_HEALTH_TIMEOUT_SECONDS = 2.0
PREFECT_HOME_ENV = "PREFECT_HOME"
PREFECT_SERVER_ANALYTICS_ENABLED_ENV = "PREFECT_SERVER_ANALYTICS_ENABLED"


class RuntimeBootstrapError(RuntimeError):
    """Raised when the configured runtime cannot be used."""


PrefectApiChecker = Callable[[str], None]
DaskSchedulerChecker = Callable[[str], object]


@dataclass(frozen=True)
class RuntimeBootstrapPlan:
    """Resolved runtime settings for a pipeline launch."""

    prefect_api_url: Optional[str]
    prefect_dashboard_url: Optional[str]
    dask_scheduler: Optional[str]
    dask_worker_count: Optional[int] = None


def prefect_api_health_url(api_url: str) -> str:
    """Return the health-check endpoint for a Prefect API URL."""
    return f"{api_url.rstrip('/')}/health"


def prefect_dashboard_url(api_url: str) -> Optional[str]:
    """Return the dashboard URL for a local/server Prefect API URL, when inferable."""
    parsed = urlsplit(api_url.rstrip("/"))
    if not parsed.scheme or not parsed.netloc:
        return None

    path = parsed.path
    dashboard_path = path[:-4] if path.endswith("/api") else path
    return urlunsplit((parsed.scheme, parsed.netloc, dashboard_path, "", ""))


def check_prefect_api(api_url: str, timeout: float = PREFECT_API_HEALTH_TIMEOUT_SECONDS) -> None:
    """Raise a helpful error if the configured Prefect API cannot be reached."""
    health_url = prefect_api_health_url(api_url)
    try:
        with urlopen(health_url, timeout=timeout) as response:
            if 200 <= response.status < 300:
                return
            raise RuntimeBootstrapError(
                f"Prefect API health check at {health_url!r} returned HTTP {response.status}"
            )
    except HTTPError as err:
        raise RuntimeBootstrapError(
            f"Prefect API health check at {health_url!r} returned HTTP {err.code}"
        ) from err
    except (TimeoutError, URLError, OSError) as err:
        raise RuntimeBootstrapError(
            "Could not reach the Prefect API at "
            f"{api_url!r}. Check PREFECT_API_URL or set "
            "cluster.prefect_api_mode = ephemeral to let Prefect use a temporary local API."
        ) from err


def resolve_prefect_api(
    execution_config: ExecutionConfig,
    checker: PrefectApiChecker = check_prefect_api,
) -> Optional[str]:
    """Resolve and validate the Prefect API URL for the configured launch mode."""
    mode = execution_config.prefect_api_mode
    api_url = execution_config.prefect_api_url

    if mode == "ephemeral":
        log.info("Using Prefect's temporary local API/server for this run.")
        return None

    if mode == "external" and not api_url:
        raise RuntimeBootstrapError(
            "cluster.prefect_api_mode = external requires cluster.prefect_api_url "
            f"or {PREFECT_API_URL_ENV}."
        )

    if api_url:
        checker(api_url)
        log.info("Using external Prefect API at %s", api_url)
        return api_url

    log.info("No Prefect API configured; Prefect may start a temporary local API/server.")
    return None


def preflight_runtime(
    execution_config: ExecutionConfig,
    prefect_api_checker: PrefectApiChecker = check_prefect_api,
    dask_scheduler_checker: DaskSchedulerChecker = check_dask_scheduler,
) -> RuntimeBootstrapPlan:
    """Validate configured Prefect and Dask runtime choices before launching the flow."""
    prefect_api_url = resolve_prefect_api(execution_config, checker=prefect_api_checker)
    dashboard_url = prefect_dashboard_url(prefect_api_url) if prefect_api_url is not None else None
    if dashboard_url is not None:
        log.info("Prefect dashboard: %s", dashboard_url)
    scheduler = execution_config.resolved_dask_scheduler()
    worker_count = None

    if execution_config.task_runner == "external_dask":
        if not scheduler:
            raise RuntimeBootstrapError(
                "prefect_task_runner = external_dask requires cluster.dask_scheduler "
                "or DASK_SCHEDULER."
            )
        worker_count = int(dask_scheduler_checker(scheduler))
        log.info("Using external Dask scheduler %s with %s worker(s).", scheduler, worker_count)
    elif execution_config.task_runner == "local_dask":
        log.info(
            "Using local Dask with %s worker(s) and %s thread(s) per worker.",
            execution_config.local_dask_worker_count,
            execution_config.local_dask_threads_per_worker,
        )
    else:
        log.info("Using synchronous Prefect task execution.")

    return RuntimeBootstrapPlan(
        prefect_api_url=prefect_api_url,
        prefect_dashboard_url=dashboard_url,
        dask_scheduler=scheduler,
        dask_worker_count=worker_count,
    )


@contextmanager
def bootstrapped_runtime(
    execution_config: ExecutionConfig,
    *,
    environ: MutableMapping[str, str] = os.environ,
    prefect_api_checker: PrefectApiChecker = check_prefect_api,
    dask_scheduler_checker: DaskSchedulerChecker = check_dask_scheduler,
) -> Iterator[RuntimeBootstrapPlan]:
    """Preflight the runtime and expose the resolved Prefect API URL for the run."""
    plan = preflight_runtime(
        execution_config,
        prefect_api_checker=prefect_api_checker,
        dask_scheduler_checker=dask_scheduler_checker,
    )
    original_api_url = environ.get(PREFECT_API_URL_ENV)
    original_prefect_home = environ.get(PREFECT_HOME_ENV)
    original_prefect_server_analytics_enabled = environ.get(PREFECT_SERVER_ANALYTICS_ENABLED_ENV)
    temporary_prefect_home = None
    try:
        environ[PREFECT_SERVER_ANALYTICS_ENABLED_ENV] = "false"
        if plan.prefect_api_url is None:
            # Override Prefect profile settings as well as shell environment.
            # An empty PREFECT_API_URL lets Prefect use its temporary local API/server.
            log.info("Ignoring any Prefect profile API URL for this run.")
            environ[PREFECT_API_URL_ENV] = ""
            temporary_prefect_home = TemporaryDirectory(prefix="rapthor-prefect-")
            environ[PREFECT_HOME_ENV] = temporary_prefect_home.name
            log.info("Using isolated temporary Prefect home for this run.")
        else:
            environ[PREFECT_API_URL_ENV] = plan.prefect_api_url
        yield plan
    finally:
        if original_prefect_home is None:
            environ.pop(PREFECT_HOME_ENV, None)
        else:
            environ[PREFECT_HOME_ENV] = original_prefect_home
        if original_api_url is None:
            environ.pop(PREFECT_API_URL_ENV, None)
        else:
            environ[PREFECT_API_URL_ENV] = original_api_url
        if original_prefect_server_analytics_enabled is None:
            environ.pop(PREFECT_SERVER_ANALYTICS_ENABLED_ENV, None)
        else:
            environ[PREFECT_SERVER_ANALYTICS_ENABLED_ENV] = (
                original_prefect_server_analytics_enabled
            )
        if temporary_prefect_home is not None:
            temporary_prefect_home.cleanup()
