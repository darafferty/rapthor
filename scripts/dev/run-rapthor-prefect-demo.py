#!/usr/bin/env python3
"""Run Rapthor through the Prefect process flow with a visible dashboard."""

from __future__ import annotations

import argparse
import configparser
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

from rapthor.execution.config import TASK_RUNNERS, ExecutionConfig
from rapthor.execution.flows.process import process_flow
from rapthor.execution.task_runner import local_cluster_kwargs
from rapthor.lib.parset import parset_read

PATH_OPTIONS = {
    "global": {
        "dir_working",
        "input_ms",
        "input_skymodel",
        "apparent_skymodel",
        "strategy",
        "input_h5parm",
        "input_fulljones_h5parm",
        "input_normalization_h5parm",
        "facet_layout",
    },
    "imaging": {
        "photometry_skymodel",
        "astrometry_skymodel",
        "normalization_skymodels",
    },
}


@dataclass
class LocalDaskDemoCluster:
    cluster: object
    client: object
    scheduler_address: str
    dashboard_url: Optional[str]
    worker_count: int

    def close(self) -> None:
        close_client = getattr(self.client, "close", None)
        if close_client is not None:
            close_client()
        close_cluster = getattr(self.cluster, "close", None)
        if close_cluster is not None:
            close_cluster()


def _default_run_dir() -> Path:
    suffix = uuid.uuid4().hex[:8]
    return Path.cwd() / "runs" / f"prefect-demo-{time.strftime('%Y%m%d-%H%M%S')}-{suffix}"


def _repo_root() -> Path:
    candidates = [Path(__file__).resolve().parents[2], Path.cwd()]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "bin").is_dir():
            return candidate
    return Path(__file__).resolve().parents[2]


def _prepend_repo_bin_to_path(repo_root: Path) -> Optional[Path]:
    bin_dir = repo_root / "bin"
    if not bin_dir.is_dir():
        return None

    bin_entry = str(bin_dir)
    existing_path = os.environ.get("PATH", "")
    entries = [entry for entry in existing_path.split(os.pathsep) if entry and entry != bin_entry]
    os.environ["PATH"] = os.pathsep.join([bin_entry, *entries])
    return bin_dir


def _resolve_demo_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return (Path.cwd() / path).resolve(strict=False)


def _working_dir_override_from_args(args: argparse.Namespace, run_dir: Path) -> Optional[Path]:
    if args.working_dir is not None:
        return _resolve_demo_path(args.working_dir)
    if args.unique_working_dir:
        return run_dir / "rapthor-work"
    return None


def _dashboard_url_from_api(api_url: str) -> str:
    return api_url.removesuffix("/api").rstrip("/")


def _dask_dashboard_url_from_address(address: Optional[str]) -> Optional[str]:
    if address is None:
        return None
    address = address.strip()
    if address in {"", "None", "none", "null", "Null"}:
        return None
    if "://" in address:
        url = address
    else:
        if address.startswith(":"):
            host = "127.0.0.1"
            port = address[1:]
        else:
            host, _, port = address.rpartition(":")
            if not host:
                host = "127.0.0.1"
            if host in {"0.0.0.0", "::", "[::]"}:
                host = "127.0.0.1"
        url = f"http://{host}:{port}" if port else f"http://{host}"
    return url.rstrip("/") if url.rstrip("/").endswith("/status") else f"{url.rstrip('/')}/status"


def _dask_dashboard_url(execution_config: ExecutionConfig) -> Optional[str]:
    if execution_config.task_runner != "local_dask":
        return None
    return _dask_dashboard_url_from_address(execution_config.dask_dashboard_address or ":8787")


def _dask_performance_report_path(
    enabled: bool,
    value: Optional[str],
    run_dir: Path,
) -> Optional[Path]:
    if not enabled:
        return None
    if value is None:
        return run_dir / "dask-performance-report.html"
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


@contextmanager
def _dask_performance_report(
    report_path: Optional[Path],
    execution_config: ExecutionConfig,
    *,
    dask_client=None,
    performance_report_factory=None,
    client_cls=None,
):
    if report_path is None:
        yield None
        return

    report_path.parent.mkdir(parents=True, exist_ok=True)
    opened_client = None
    if performance_report_factory is None or client_cls is None:
        try:
            from dask.distributed import Client, performance_report
        except ImportError as err:
            raise RuntimeError(
                "Writing a Dask performance report requires dask.distributed."
            ) from err
        performance_report_factory = performance_report_factory or performance_report
        client_cls = client_cls or Client

    if dask_client is None:
        scheduler = execution_config.resolved_dask_scheduler()
        if not scheduler:
            raise RuntimeError(
                "--dask-performance-report requires a persistent local or external "
                "Dask scheduler. Keep --start-dask enabled or pass --dask-scheduler."
            )
        opened_client = client_cls(scheduler)

    try:
        with performance_report_factory(filename=str(report_path)):
            yield report_path
    finally:
        if opened_client is not None:
            opened_client.close()


def _api_url_for_local_server(port: int) -> str:
    return f"http://127.0.0.1:{port}/api"


def _health_url(api_url: str) -> str:
    return f"{api_url.rstrip('/')}/health"


def _is_prefect_healthy(api_url: str, timeout: float = 2.0) -> bool:
    try:
        with urlopen(_health_url(api_url), timeout=timeout) as response:
            return 200 <= response.status < 300
    except URLError:
        return False
    except TimeoutError:
        return False


def _wait_for_prefect(api_url: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _is_prefect_healthy(api_url):
            return
        time.sleep(1)
    raise TimeoutError(f"Prefect API did not become healthy at {_health_url(api_url)}")


def _start_prefect_server(
    host: str,
    port: int,
    run_dir: Path,
    api_url: str,
    timeout_seconds: int,
) -> Optional[subprocess.Popen]:
    if _is_prefect_healthy(api_url):
        print(f"Using existing Prefect server at {api_url}")
        return None

    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "prefect-server.log"
    print(f"Starting Prefect server on {host}:{port}")
    print(f"Prefect server log: {log_file}")

    log_handle = log_file.open("w")
    try:
        server = subprocess.Popen(
            ["prefect", "server", "start", "--host", host, "--port", str(port)],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_handle.close()
    try:
        _wait_for_prefect(api_url, timeout_seconds)
    except Exception:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        raise
    return server


def _is_empty_path_value(value: str) -> bool:
    return value.strip() in {"", "None", "none", "null", "Null"}


def _resolve_path_token(value: str, base_dir: Path) -> str:
    token = value.strip()
    if _is_empty_path_value(token):
        return value
    if "://" in token:
        return token
    path = Path(token).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve(strict=False))


def _resolve_path_value(value: str, base_dir: Path) -> str:
    stripped = value.strip()
    if _is_empty_path_value(stripped):
        return value
    if stripped.startswith("[") and stripped.endswith("]"):
        resolved = [
            _resolve_path_token(token, base_dir)
            for token in stripped.strip("[]").split(",")
            if token.strip()
        ]
        return f"[{', '.join(resolved)}]"
    return _resolve_path_token(value, base_dir)


def _materialize_parset_paths(
    parset_file: Path,
    run_dir: Path,
    working_dir_override: Optional[Path] = None,
) -> Path:
    parser = configparser.ConfigParser(interpolation=None)
    with parset_file.open() as handle:
        parser.read_file(handle)

    base_dir = Path.cwd()
    for section, options in PATH_OPTIONS.items():
        if not parser.has_section(section):
            continue
        for option in options:
            if parser.has_option(section, option):
                parser.set(
                    section, option, _resolve_path_value(parser.get(section, option), base_dir)
                )

    if working_dir_override is not None:
        if not parser.has_section("global"):
            parser.add_section("global")
        parser.set("global", "dir_working", str(working_dir_override))

    materialized = run_dir / f"{parset_file.stem}.materialized.parset"
    with materialized.open("w") as handle:
        parser.write(handle)
    return materialized


def _write_runtime_cluster_overrides(
    parset_file: Path,
    run_dir: Path,
    overrides: dict[str, object],
) -> Path:
    parser = configparser.ConfigParser(interpolation=None)
    with parset_file.open() as handle:
        parser.read_file(handle)

    section = "cluster" if parser.has_section("cluster") else "cluster_specific"
    if not parser.has_section(section):
        parser.add_section(section)

    for key, value in overrides.items():
        parser.set(section, key, "" if value is None else str(value))

    runtime_parset = run_dir / f"{parset_file.stem}.runtime.parset"
    with runtime_parset.open("w") as handle:
        parser.write(handle)
    return runtime_parset


def _execution_config_from_args(parset_file: Path, args: argparse.Namespace) -> ExecutionConfig:
    config = ExecutionConfig.from_parset(parset_read(parset_file))
    overrides = {}

    if args.task_runner is not None:
        overrides["task_runner"] = args.task_runner
    if args.dask_scheduler is not None:
        overrides["dask_scheduler"] = args.dask_scheduler
    if args.dask_dashboard_address is not None:
        overrides["dask_dashboard_address"] = args.dask_dashboard_address
    if args.stream_output is not None:
        overrides["stream_output"] = args.stream_output
    if args.max_nodes is not None:
        overrides["max_nodes"] = args.max_nodes
    if args.cpus_per_task is not None:
        overrides["cpus_per_task"] = args.cpus_per_task

    return replace(config, **overrides) if overrides else config


def _start_local_dask_cluster(execution_config: ExecutionConfig) -> LocalDaskDemoCluster:
    try:
        from dask.distributed import Client, LocalCluster
    except ImportError as err:
        raise RuntimeError(
            "Starting the local Dask dashboard requires dask.distributed. "
            "Install the Prefect/Dask runtime dependencies first."
        ) from err

    cluster_kwargs = local_cluster_kwargs(execution_config)
    cluster_kwargs.setdefault("dashboard_address", ":8787")
    cluster = LocalCluster(**cluster_kwargs)
    try:
        client = Client(cluster)
        worker_count = execution_config.local_dask_worker_count
        client.wait_for_workers(worker_count, timeout="60s")
    except Exception:
        cluster.close()
        raise

    return LocalDaskDemoCluster(
        cluster=cluster,
        client=client,
        scheduler_address=cluster.scheduler_address,
        dashboard_url=cluster.dashboard_link,
        worker_count=worker_count,
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start or attach to a Prefect server and run Rapthor through "
            "rapthor.execution.flows.process.process_flow()."
        )
    )
    parser.add_argument("parset", type=Path, help="Rapthor parset file to run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(os.environ["RAPTHOR_PREFECT_DEMO_DIR"])
        if "RAPTHOR_PREFECT_DEMO_DIR" in os.environ
        else None,
        help="Directory for demo logs. Defaults to ./runs/prefect-demo-<timestamp>.",
    )
    parser.add_argument(
        "--unique-working-dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Override global.dir_working to a directory inside this demo run. "
            "Defaults to true so repeated demo runs do not reuse pipeline state."
        ),
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        help=(
            "Explicit Rapthor global.dir_working override. Defaults to "
            "<run-dir>/rapthor-work when --unique-working-dir is enabled."
        ),
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("PREFECT_HOST", "0.0.0.0"),
        help="Bind host for a new Prefect server. Defaults to PREFECT_HOST or 0.0.0.0.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PREFECT_PORT", "4200")),
        help="Port for a new Prefect server. Defaults to PREFECT_PORT or 4200.",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("PREFECT_API_URL"),
        help="Prefect API URL to use. Defaults to PREFECT_API_URL or localhost:<port>/api.",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Require an already-running Prefect API instead of starting one.",
    )
    parser.add_argument(
        "--materialize-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write a temporary parset with relative path options resolved to absolute "
            "paths before running. Defaults to true."
        ),
    )
    parser.add_argument(
        "--server-startup-timeout",
        type=int,
        default=60,
        help="Seconds to wait for a newly started Prefect server.",
    )
    parser.add_argument(
        "--task-runner",
        choices=TASK_RUNNERS,
        help="Override cluster.prefect_task_runner from the parset.",
    )
    parser.add_argument(
        "--dask-scheduler",
        help="Override the Dask scheduler address, e.g. tcp://scheduler:8786.",
    )
    parser.add_argument(
        "--dask-dashboard-address",
        default=os.environ.get("DASK_DASHBOARD_ADDRESS"),
        help=(
            "Dashboard address for the local Dask cluster, e.g. :8787 or "
            "127.0.0.1:8787. Applies only with --task-runner local_dask."
        ),
    )
    parser.add_argument(
        "--dask-performance-report",
        action="store_true",
        help=(
            "Write a Dask performance report HTML file in the demo run directory. "
            "Use --dask-performance-report-path to choose a different path."
        ),
    )
    parser.add_argument(
        "--dask-performance-report-path",
        metavar="PATH",
        help="Path for --dask-performance-report output.",
    )
    parser.add_argument(
        "--start-dask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Start a persistent local Dask cluster when the effective task runner "
            "is local_dask. Defaults to true so the dashboard shows workers and "
            "task stream activity for the whole run."
        ),
    )
    parser.add_argument(
        "--stream-output",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Stream external command stdout/stderr into Prefect task logs. "
            "Defaults to the parset's cluster.prefect_stream_output value."
        ),
    )
    parser.add_argument("--max-nodes", type=int, help="Override cluster.max_nodes.")
    parser.add_argument("--cpus-per-task", type=int, help="Override cluster.cpus_per_task.")
    parser.add_argument(
        "--logging-level",
        default=os.environ.get("RAPTHOR_LOGGING_LEVEL", "info"),
        help="Rapthor logging level passed to process_flow().",
    )
    parser.add_argument(
        "--keep-server-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep a server started by this helper running if Rapthor fails. "
            "Defaults to true so the dashboard remains available for inspection."
        ),
    )
    parser.add_argument(
        "--keep-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep a server started by this helper running after Rapthor finishes. "
            "Defaults to true so completed runs remain visible in the dashboard."
        ),
    )
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    repo_bin = _prepend_repo_bin_to_path(_repo_root())
    parset_file = args.parset.resolve()
    if not parset_file.exists():
        sys.stderr.write(f"Parset does not exist: {parset_file}\n")
        return 2

    run_dir = _resolve_demo_path(args.run_dir) if args.run_dir is not None else _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    working_dir_override = _working_dir_override_from_args(args, run_dir)
    run_parset_file = (
        _materialize_parset_paths(
            parset_file,
            run_dir,
            working_dir_override=working_dir_override,
        )
        if args.materialize_paths or working_dir_override is not None
        else parset_file
    )

    api_url = args.api_url or _api_url_for_local_server(args.port)
    os.environ["PREFECT_API_URL"] = api_url

    server = None
    dask_cluster = None
    failed = False
    try:
        if args.no_start_server:
            _wait_for_prefect(api_url, timeout_seconds=5)
            print(f"Using existing Prefect server at {api_url}")
        else:
            server = _start_prefect_server(
                args.host,
                args.port,
                run_dir,
                api_url,
                args.server_startup_timeout,
            )

        print(f"Prefect dashboard: {_dashboard_url_from_api(api_url)}")
        print(f"Rapthor parset: {parset_file}")
        if run_parset_file != parset_file:
            print(f"Materialized parset: {run_parset_file}")
        print(f"Demo run directory: {run_dir}")
        if repo_bin is not None:
            print(f"Rapthor helper scripts: {repo_bin}")
        if working_dir_override is not None:
            print(f"Rapthor working directory: {working_dir_override}")

        execution_config = _execution_config_from_args(run_parset_file, args)
        if args.start_dask and execution_config.task_runner == "local_dask":
            print("Starting local Dask cluster for dashboard monitoring.")
            cluster_config = execution_config
            if cluster_config.dask_dashboard_address is None:
                cluster_config = replace(cluster_config, dask_dashboard_address=":8787")
            dask_cluster = _start_local_dask_cluster(cluster_config)
            execution_config = replace(
                cluster_config,
                task_runner="external_dask",
                dask_scheduler=dask_cluster.scheduler_address,
            )
            run_parset_file = _write_runtime_cluster_overrides(
                run_parset_file,
                run_dir,
                {
                    "prefect_task_runner": execution_config.task_runner,
                    "dask_scheduler": execution_config.dask_scheduler,
                    "dask_dashboard_address": execution_config.dask_dashboard_address,
                },
            )
            print(f"Runtime parset: {run_parset_file}")

        print(
            "Execution config: "
            f"task_runner={execution_config.task_runner}, "
            f"dask_scheduler={execution_config.resolved_dask_scheduler()}, "
            f"dask_dashboard_address={execution_config.dask_dashboard_address}, "
            f"max_nodes={execution_config.max_nodes}, "
            f"cpus_per_task={execution_config.cpus_per_task}"
        )
        dask_dashboard_url = (
            dask_cluster.dashboard_url
            if dask_cluster is not None
            else _dask_dashboard_url(execution_config)
        )
        if dask_dashboard_url is not None:
            print(f"Dask dashboard: {dask_dashboard_url}")
            if dask_cluster is not None:
                print(
                    "Dask workers: "
                    f"{dask_cluster.worker_count} connected to {dask_cluster.scheduler_address}"
                )
        elif execution_config.task_runner == "external_dask":
            print("Dask dashboard: use the dashboard for the external Dask scheduler.")

        performance_report_path = _dask_performance_report_path(
            args.dask_performance_report,
            args.dask_performance_report_path,
            run_dir,
        )
        if performance_report_path is not None:
            print(f"Dask performance report: {performance_report_path}")

        with _dask_performance_report(
            performance_report_path,
            execution_config,
            dask_client=None if dask_cluster is None else dask_cluster.client,
        ):
            process_flow(
                run_parset_file,
                logging_level=args.logging_level,
                execution_config=execution_config,
            )
        if performance_report_path is not None:
            print(f"Wrote Dask performance report: {performance_report_path}")
        print("Rapthor Prefect demo run finished.")
        return 0
    except Exception:
        failed = True
        raise
    finally:
        if dask_cluster is not None:
            print("Stopping local Dask cluster.")
            dask_cluster.close()
        if server is not None:
            keep_server = args.keep_server or (failed and args.keep_server_on_failure)
            if keep_server:
                status = "failed" if failed else "finished"
                print(f"Rapthor {status}; leaving the Prefect server running.")
                print(f"Prefect dashboard: {_dashboard_url_from_api(api_url)}")
                print(f"Prefect server PID: {server.pid}")
                print(
                    "Stop it manually when finished, for example with: "
                    "pkill -f 'prefect server start'"
                )
            else:
                print("Stopping Prefect server.")
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
