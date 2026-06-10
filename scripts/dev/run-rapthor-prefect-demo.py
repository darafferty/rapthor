#!/usr/bin/env python3
"""Run Rapthor through the Prefect process flow with a visible dashboard."""

from __future__ import annotations

import argparse
import configparser
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

from rapthor.execution.config import TASK_RUNNERS, ExecutionConfig
from rapthor.execution.flows.process import process_flow
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


def _default_run_dir() -> Path:
    return Path.cwd() / "runs" / f"prefect-demo-{time.strftime('%Y%m%d-%H%M%S')}"


def _dashboard_url_from_api(api_url: str) -> str:
    return api_url.removesuffix("/api").rstrip("/")


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


def _materialize_parset_paths(parset_file: Path, run_dir: Path) -> Path:
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

    materialized = run_dir / f"{parset_file.stem}.materialized.parset"
    with materialized.open("w") as handle:
        parser.write(handle)
    return materialized


def _execution_config_from_args(parset_file: Path, args: argparse.Namespace) -> ExecutionConfig:
    config = ExecutionConfig.from_parset(parset_read(parset_file))
    overrides = {}

    if args.task_runner is not None:
        overrides["task_runner"] = args.task_runner
    if args.dask_scheduler is not None:
        overrides["dask_scheduler"] = args.dask_scheduler
    if args.stream_output is not None:
        overrides["stream_output"] = args.stream_output
    if args.max_nodes is not None:
        overrides["max_nodes"] = args.max_nodes
    if args.cpus_per_task is not None:
        overrides["cpus_per_task"] = args.cpus_per_task

    return replace(config, **overrides) if overrides else config


def _parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    parset_file = args.parset.resolve()
    if not parset_file.exists():
        sys.stderr.write(f"Parset does not exist: {parset_file}\n")
        return 2

    run_dir = (args.run_dir or _default_run_dir()).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_parset_file = (
        _materialize_parset_paths(parset_file, run_dir) if args.materialize_paths else parset_file
    )

    api_url = args.api_url or _api_url_for_local_server(args.port)
    os.environ["PREFECT_API_URL"] = api_url

    server = None
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

        execution_config = _execution_config_from_args(run_parset_file, args)
        print(
            "Execution config: "
            f"task_runner={execution_config.task_runner}, "
            f"dask_scheduler={execution_config.resolved_dask_scheduler()}, "
            f"max_nodes={execution_config.max_nodes}, "
            f"cpus_per_task={execution_config.cpus_per_task}"
        )

        process_flow(
            run_parset_file,
            logging_level=args.logging_level,
            execution_config=execution_config,
        )
        print("Rapthor Prefect demo run finished.")
        return 0
    except Exception:
        failed = True
        raise
    finally:
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
