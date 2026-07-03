"""Benchmark scenario definitions and report helpers for Rapthor runs."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class BenchmarkScenario:
    """A reproducible Rapthor benchmark scenario."""

    scenario_id: str
    description: str
    parset: str
    local_dask_workers: int
    cpus_per_task: int
    max_threads: int
    command_profile: str = "time"

    def command(self, repo_root: Path, run_dir: Path) -> list[str]:
        """Return the command used to run this scenario."""
        return [
            sys.executable,
            str(repo_root / "scripts" / "dev" / "run-rapthor-prefect-demo.py"),
            str(repo_root / self.parset),
            "--run-dir",
            str(run_dir),
            "--task-runner",
            "local_dask",
            "--local-dask-workers",
            str(self.local_dask_workers),
            "--cpus-per-task",
            str(self.cpus_per_task),
            "--max-threads",
            str(self.max_threads),
            "--command-profile",
            self.command_profile,
            "--dask-performance-report",
            "--dask-performance-report-path",
            str(run_dir / "dask-performance-report.html"),
            "--no-keep-server",
            "--no-keep-server-on-failure",
        ]


@dataclass(frozen=True)
class CommandMetric:
    """A single external command timing parsed from ``logs/commands.jsonl``."""

    name: str
    operation: Optional[str]
    status: Optional[str]
    duration_seconds: float


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Recorded outcome for one benchmark repetition."""

    scenario_id: str
    repetition: int
    run_dir: str
    returncode: int
    wall_seconds: float
    command_count: int
    command_seconds: float
    dask_performance_report: Optional[str] = None


def default_benchmark_scenarios() -> tuple[BenchmarkScenario, ...]:
    """Return the committed baseline scenarios from ``PLAN.md``."""
    return (
        BenchmarkScenario(
            scenario_id="quick-demo",
            description="Tiny fixture for startup overhead and CLI/runtime cost.",
            parset="examples/prefect_demo.parset",
            local_dask_workers=1,
            cpus_per_task=4,
            max_threads=4,
        ),
        BenchmarkScenario(
            scenario_id="rich-demo",
            description="Generated rich demo for representative Prefect/Dask graph shape.",
            parset="examples/generated/prefect_demo_rich/prefect_demo_rich.parset",
            local_dask_workers=2,
            cpus_per_task=4,
            max_threads=4,
        ),
    )


def benchmark_scenarios_by_id() -> dict[str, BenchmarkScenario]:
    """Return default scenarios keyed by scenario id."""
    return {scenario.scenario_id: scenario for scenario in default_benchmark_scenarios()}


def parse_command_log(command_log: Path) -> list[CommandMetric]:
    """Parse Rapthor command timing records from ``logs/commands.jsonl``."""
    metrics = []
    if not command_log.exists():
        return metrics

    with command_log.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON in {command_log}:{line_number}") from err
            metrics.append(
                CommandMetric(
                    name=str(record.get("name") or _command_name(record.get("command"))),
                    operation=_optional_str(record.get("operation")),
                    status=_optional_str(record.get("status")),
                    duration_seconds=float(record.get("duration_seconds") or 0.0),
                )
            )
    return metrics


def benchmark_run_result(
    *,
    scenario_id: str,
    repetition: int,
    run_dir: Path,
    returncode: int,
    wall_seconds: float,
) -> BenchmarkRunResult:
    """Build a run result from a completed run directory."""
    command_metrics = parse_command_log(run_dir / "rapthor-work" / "logs" / "commands.jsonl")
    dask_report = run_dir / "dask-performance-report.html"
    return BenchmarkRunResult(
        scenario_id=scenario_id,
        repetition=repetition,
        run_dir=str(run_dir),
        returncode=returncode,
        wall_seconds=wall_seconds,
        command_count=len(command_metrics),
        command_seconds=sum(metric.duration_seconds for metric in command_metrics),
        dask_performance_report=str(dask_report) if dask_report.exists() else None,
    )


def summarize_benchmark_runs(results: Iterable[BenchmarkRunResult]) -> dict[str, object]:
    """Summarize benchmark repetitions by scenario."""
    grouped: dict[str, list[BenchmarkRunResult]] = {}
    for result in results:
        grouped.setdefault(result.scenario_id, []).append(result)

    scenarios = {}
    for scenario_id, scenario_results in sorted(grouped.items()):
        wall_seconds = [result.wall_seconds for result in scenario_results]
        command_seconds = [result.command_seconds for result in scenario_results]
        scenarios[scenario_id] = {
            "runs": len(scenario_results),
            "returncodes": [result.returncode for result in scenario_results],
            "wall_seconds": _stats(wall_seconds),
            "command_seconds": _stats(command_seconds),
            "command_count": _stats([result.command_count for result in scenario_results]),
            "run_dirs": [result.run_dir for result in scenario_results],
            "dask_performance_reports": [
                result.dask_performance_report
                for result in scenario_results
                if result.dask_performance_report
            ],
        }

    return {"scenarios": scenarios}


def render_markdown_report(summary: Mapping[str, object]) -> str:
    """Render a compact Markdown benchmark report."""
    lines = [
        "# Rapthor Benchmark Baseline",
        "",
        "| Scenario | Runs | Return Codes | Wall Median (s) | Wall Min-Max (s) | "
        "Command Median (s) |",
        "| --- | ---: | --- | ---: | --- | ---: |",
    ]
    scenarios = summary.get("scenarios", {})
    if not isinstance(scenarios, Mapping):
        raise TypeError("summary['scenarios'] must be a mapping")

    for scenario_id, data in sorted(scenarios.items()):
        if not isinstance(data, Mapping):
            raise TypeError("scenario summaries must be mappings")
        wall = data["wall_seconds"]
        command = data["command_seconds"]
        lines.append(
            "| {scenario} | {runs} | {returncodes} | {wall_median:.3f} | "
            "{wall_min:.3f}-{wall_max:.3f} | {command_median:.3f} |".format(
                scenario=scenario_id,
                runs=data["runs"],
                returncodes=", ".join(map(str, data["returncodes"])),
                wall_median=wall["median"],
                wall_min=wall["min"],
                wall_max=wall["max"],
                command_median=command["median"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_scenario_once(
    scenario: BenchmarkScenario,
    *,
    repetition: int,
    run_root: Path,
    repo_root: Path,
    command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
) -> BenchmarkRunResult:
    """Run one scenario repetition and return a summarized result."""
    run_dir = run_root / scenario.scenario_id / f"rep-{repetition:02d}"
    if run_dir.exists():
        raise FileExistsError(f"Benchmark run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    command = scenario.command(repo_root, run_dir)
    command_record = {
        "scenario_id": scenario.scenario_id,
        "repetition": repetition,
        "command": command,
        "run_dir": str(run_dir),
    }
    (run_dir / "benchmark-command.json").write_text(
        json.dumps(command_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    started = monotonic()
    completed = command_runner(command, cwd=repo_root, check=False)
    wall_seconds = monotonic() - started
    result = benchmark_run_result(
        scenario_id=scenario.scenario_id,
        repetition=repetition,
        run_dir=run_dir,
        returncode=int(completed.returncode),
        wall_seconds=wall_seconds,
    )
    (run_dir / "benchmark-result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def write_summary_artifacts(
    results: Sequence[BenchmarkRunResult],
    *,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, object]:
    """Write JSON and Markdown benchmark summary artifacts."""
    summary = summarize_benchmark_runs(results)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_markdown.write_text(render_markdown_report(summary), encoding="utf-8")
    return summary


def failed_benchmark_runs(results: Iterable[BenchmarkRunResult]) -> list[BenchmarkRunResult]:
    """Return benchmark repetitions whose command exited unsuccessfully."""
    return [result for result in results if result.returncode != 0]


def format_failed_benchmark_runs(results: Iterable[BenchmarkRunResult]) -> str:
    """Format failed benchmark repetitions for CI and terminal output."""
    failed = failed_benchmark_runs(results)
    if not failed:
        return ""

    lines = ["Benchmark repetitions failed:"]
    for result in failed:
        lines.append(
            f"- {result.scenario_id} repetition {result.repetition}: "
            f"return code {result.returncode}, run directory {result.run_dir}"
        )
    return "\n".join(lines)


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _optional_str(value: object) -> Optional[str]:
    if value in (None, "", "None"):
        return None
    return str(value)


def _command_name(command: object) -> str:
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        if command:
            return Path(str(command[0])).name
    return "unknown"
