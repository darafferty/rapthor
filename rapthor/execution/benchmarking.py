"""Benchmark scenario definitions and report helpers for Rapthor runs."""

from __future__ import annotations

import configparser
import json
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from typing import Optional

METADATA_LABELS = {
    "container_image": "Container image",
    "gitlab_commit_sha": "GitLab commit SHA",
    "gitlab_job_id": "GitLab job ID",
    "gitlab_job_url": "GitLab job URL",
    "gitlab_pipeline_id": "GitLab pipeline ID",
    "gitlab_pipeline_url": "GitLab pipeline URL",
    "gitlab_project_url": "GitLab project URL",
    "gitlab_ref_name": "GitLab ref",
    "runner_description": "GitLab runner",
    "runner_id": "GitLab runner ID",
}


@dataclass(frozen=True)
class ParsetOverride:
    """One scenario-specific parset option override."""

    section: str
    option: str
    value: str


@dataclass(frozen=True)
class BenchmarkScenario:
    """A reproducible Rapthor benchmark scenario."""

    scenario_id: str
    description: str
    parset: str
    local_dask_workers: int
    cpus_per_task: int
    max_threads: int
    filter_skymodel_ncores: Optional[int] = None
    command_profile: str = "time"
    parset_overrides: tuple[ParsetOverride, ...] = ()

    def command(self, repo_root: Path, run_dir: Path) -> list[str]:
        """Return the command used to run this scenario."""
        parset_path = self.parset_path_for_run(repo_root, run_dir)
        return [
            sys.executable,
            str(repo_root / "scripts" / "dev" / "run-rapthor-prefect-demo.py"),
            str(parset_path),
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
            *(
                []
                if self.filter_skymodel_ncores is None
                else ["--filter-skymodel-ncores", str(self.filter_skymodel_ncores)]
            ),
            "--command-profile",
            self.command_profile,
            "--dask-performance-report",
            "--dask-performance-report-path",
            str(run_dir / "dask-performance-report.html"),
            "--no-keep-server",
            "--no-keep-server-on-failure",
        ]

    def parset_path_for_run(self, repo_root: Path, run_dir: Path) -> Path:
        """Return a scenario parset, materializing overrides when needed."""
        base_parset = repo_root / self.parset
        if not self.parset_overrides:
            return base_parset

        parser = configparser.ConfigParser(interpolation=None)
        with base_parset.open(encoding="utf-8") as handle:
            parser.read_file(handle)
        for override in self.parset_overrides:
            if not parser.has_section(override.section):
                parser.add_section(override.section)
            parser.set(override.section, override.option, override.value)

        scenario_parset = run_dir / f"{Path(self.parset).stem}.{self.scenario_id}.parset"
        scenario_parset.parent.mkdir(parents=True, exist_ok=True)
        with scenario_parset.open("w", encoding="utf-8") as handle:
            parser.write(handle)
        return scenario_parset


@dataclass(frozen=True)
class CommandMetric:
    """A single external command timing parsed from ``logs/commands.jsonl``."""

    name: str
    operation: Optional[str]
    status: Optional[str]
    duration_seconds: float


@dataclass(frozen=True)
class OperationTimingMetric:
    """Operation elapsed timing compared with profiled external commands."""

    operation: str
    elapsed_seconds: float
    command_count: int = 0
    command_seconds: float = 0.0
    operation_minus_command_seconds: Optional[float] = None


@dataclass(frozen=True)
class DaskTaskGroupMetric:
    """Aggregated task-stream timing for one Dask task name."""

    name: str
    count: int
    total_seconds: float
    median_seconds: float
    max_seconds: float


@dataclass(frozen=True)
class DaskPerformanceMetric:
    """Summary values parsed from a Dask performance report."""

    duration_seconds: Optional[float] = None
    compute_seconds: Optional[float] = None
    task_count: Optional[int] = None
    workers: Optional[int] = None
    threads: Optional[int] = None
    memory: Optional[str] = None
    task_groups: tuple[DaskTaskGroupMetric, ...] = ()

    @property
    def duration_minus_compute_seconds(self) -> Optional[float]:
        if self.duration_seconds is None or self.compute_seconds is None:
            return None
        return self.duration_seconds - self.compute_seconds


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
    dask_duration_seconds: Optional[float] = None
    dask_compute_seconds: Optional[float] = None
    dask_duration_minus_compute_seconds: Optional[float] = None
    dask_task_count: Optional[int] = None
    dask_workers: Optional[int] = None
    dask_threads: Optional[int] = None
    dask_memory: Optional[str] = None
    dask_task_groups: tuple[DaskTaskGroupMetric, ...] = ()
    operation_timings: tuple[OperationTimingMetric, ...] = ()
    command_timings: tuple[CommandMetric, ...] = ()


def default_benchmark_scenarios() -> tuple[BenchmarkScenario, ...]:
    """Return the default baseline scenario from ``PLAN.md``."""
    return (
        BenchmarkScenario(
            scenario_id="ci-benchmark",
            description=(
                "Generated CI benchmark exercising DI phase, DD phase/faceting, "
                "legacy DD default solves, full-Jones, imaging, mosaicking, "
                "and source filtering."
            ),
            parset="examples/generated/prefect_demo_rich/prefect_demo_benchmark.parset",
            local_dask_workers=2,
            cpus_per_task=30,
            max_threads=30,
        ),
        BenchmarkScenario(
            scenario_id="ci-benchmark-image-products",
            description=(
                "Generated CI benchmark variant exercising image post-processing: "
                "flux-scale normalization, final full-Stokes image cubes, cube "
                "catalog creation, restoration, and final-image compression."
            ),
            parset="examples/generated/prefect_demo_rich/prefect_demo_benchmark.parset",
            local_dask_workers=2,
            cpus_per_task=30,
            max_threads=30,
            parset_overrides=(
                ParsetOverride(
                    "global",
                    "strategy",
                    "examples/generated/prefect_demo_rich/"
                    "prefect_demo_benchmark_normalize_strategy.py",
                ),
                ParsetOverride(
                    "imaging",
                    "normalization_skymodels",
                    "["
                    "examples/generated/prefect_demo_rich/"
                    "prefect_demo_rich_reference_120mhz.txt, "
                    "examples/generated/prefect_demo_rich/"
                    "prefect_demo_rich_reference_160mhz.txt"
                    "]",
                ),
                ParsetOverride(
                    "imaging",
                    "normalization_reference_frequencies",
                    "[120000000.0, 160000000.0]",
                ),
                ParsetOverride("imaging", "make_quv_images", "True"),
                ParsetOverride("imaging", "disable_iquv_clean", "True"),
                ParsetOverride("imaging", "save_image_cube", "True"),
                ParsetOverride("imaging", "image_cube_stokes_list", "[I, Q, U, V]"),
                ParsetOverride("imaging", "compress_final_images", "True"),
            ),
        ),
        BenchmarkScenario(
            scenario_id="ci-benchmark-predict-chunks",
            description=(
                "Generated CI benchmark variant forcing observation chunking so "
                "predict model-data and post-processing dependencies can be "
                "measured across multiple time chunks."
            ),
            parset="examples/generated/prefect_demo_rich/prefect_demo_benchmark.parset",
            local_dask_workers=2,
            cpus_per_task=30,
            max_threads=30,
            parset_overrides=(ParsetOverride("cluster", "max_nodes", "2"),),
        ),
        BenchmarkScenario(
            scenario_id="ci-benchmark-calibration-postprocess",
            description=(
                "Generated CI benchmark variant isolating calibration solution "
                "post-processing: DD phase solves, slow-gain processing, solution "
                "plotting, and h5parm combination without imaging."
            ),
            parset="examples/generated/prefect_demo_rich/prefect_demo_benchmark.parset",
            local_dask_workers=2,
            cpus_per_task=30,
            max_threads=30,
            parset_overrides=(
                ParsetOverride(
                    "global",
                    "strategy",
                    "examples/generated/prefect_demo_rich/"
                    "prefect_demo_benchmark_calibration_postprocess_strategy.py",
                ),
            ),
        ),
        BenchmarkScenario(
            scenario_id="ci-benchmark-wsclean-predict",
            description=(
                "Generated CI benchmark variant exercising the WSClean-predict "
                "calibration path and its prediction post-processing."
            ),
            parset="examples/generated/prefect_demo_rich/prefect_demo_benchmark.parset",
            local_dask_workers=2,
            cpus_per_task=30,
            max_threads=30,
            parset_overrides=(ParsetOverride("calibration", "use_wsclean_predict", "True"),),
        ),
        BenchmarkScenario(
            scenario_id="ci-benchmark-many-sector-mosaic",
            description=(
                "Generated CI benchmark variant using quadrant-balanced sky "
                "directions to exercise multiple image sectors and heavier "
                "mosaic regridding/assembly work."
            ),
            parset=(
                "examples/generated/prefect_demo_rich/prefect_demo_multisector_benchmark.parset"
            ),
            local_dask_workers=2,
            cpus_per_task=30,
            max_threads=30,
        ),
        BenchmarkScenario(
            scenario_id="ci-benchmark-many-sector-mosaic-sparse-fallback",
            description=(
                "Generated CI benchmark variant using the same many-sector "
                "mosaic inputs but forcing sparse FITS model-mosaic regridding "
                "instead of WSClean-rendered model mosaics."
            ),
            parset=(
                "examples/generated/prefect_demo_rich/prefect_demo_multisector_benchmark.parset"
            ),
            local_dask_workers=2,
            cpus_per_task=30,
            max_threads=30,
            parset_overrides=(ParsetOverride("imaging", "model_mosaic_method", "sparse_fits"),),
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


def parse_operation_log(operation_log: Path) -> list[OperationTimingMetric]:
    """Parse Rapthor operation boundary timings from ``rapthor.log``."""
    if not operation_log.exists():
        return []

    timings = []
    seen = set()
    for line in operation_log.read_text(encoding="utf-8", errors="replace").splitlines():
        clean_line = _strip_ansi(line)
        match = re.search(
            r"rapthor:(?P<operation>[\w_]+).*Time for operation:\s*"
            r"(?P<duration>(?:\d+\s+days?,\s+)?\d+:\d{2}:\d{2}(?:\.\d+)?)",
            clean_line,
        )
        if not match:
            continue
        key = (match.group("operation"), match.group("duration"))
        if key in seen:
            continue
        seen.add(key)
        timings.append(
            OperationTimingMetric(
                operation=match.group("operation"),
                elapsed_seconds=_duration_text_to_seconds(match.group("duration")),
            )
        )
    return timings


def parse_dask_performance_report(report_path: Path) -> Optional[DaskPerformanceMetric]:
    """Parse summary and task-stream timing from a Dask performance report."""
    if not report_path.exists():
        return None

    text = unescape(report_path.read_text(encoding="utf-8", errors="replace"))
    return DaskPerformanceMetric(
        duration_seconds=_optional_float_match(r"Duration:\s*([0-9.]+)\s*s", text),
        compute_seconds=_optional_float_match(r"compute time:\s*([0-9.]+)\s*s", text),
        task_count=_optional_int_match(r"number of tasks:\s*([0-9]+)", text),
        workers=_optional_int_match(r"Workers:\s*([0-9]+)", text),
        threads=_optional_int_match(r"Threads:\s*([0-9]+)", text),
        memory=_optional_text_match(r"Memory:\s*([^<\n]+)", text),
        task_groups=_parse_dask_task_groups(text),
    )


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
    operation_timings = _combine_operation_timings(
        parse_operation_log(run_dir / "rapthor-work" / "logs" / "rapthor.log"),
        command_metrics,
    )
    dask_report = run_dir / "dask-performance-report.html"
    dask_metrics = parse_dask_performance_report(dask_report)
    return BenchmarkRunResult(
        scenario_id=scenario_id,
        repetition=repetition,
        run_dir=str(run_dir),
        returncode=returncode,
        wall_seconds=wall_seconds,
        command_count=len(command_metrics),
        command_seconds=sum(metric.duration_seconds for metric in command_metrics),
        dask_performance_report=str(dask_report) if dask_report.exists() else None,
        dask_duration_seconds=dask_metrics.duration_seconds if dask_metrics else None,
        dask_compute_seconds=dask_metrics.compute_seconds if dask_metrics else None,
        dask_duration_minus_compute_seconds=(
            dask_metrics.duration_minus_compute_seconds if dask_metrics else None
        ),
        dask_task_count=dask_metrics.task_count if dask_metrics else None,
        dask_workers=dask_metrics.workers if dask_metrics else None,
        dask_threads=dask_metrics.threads if dask_metrics else None,
        dask_memory=dask_metrics.memory if dask_metrics else None,
        dask_task_groups=dask_metrics.task_groups if dask_metrics else (),
        operation_timings=operation_timings,
        command_timings=tuple(command_metrics),
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
        dask_summary = _summarize_dask_metrics(scenario_results)
        if dask_summary:
            scenarios[scenario_id]["dask"] = dask_summary
        operation_summary = _summarize_operation_timings(scenario_results)
        if operation_summary:
            scenarios[scenario_id]["operations"] = operation_summary
        command_summary = _summarize_command_timings(scenario_results)
        if command_summary:
            scenarios[scenario_id]["commands"] = command_summary

    return {"scenarios": scenarios}


def render_markdown_report(summary: Mapping[str, object]) -> str:
    """Render a compact Markdown benchmark report."""
    lines = [
        "# Rapthor Benchmark Baseline",
        "",
    ]
    metadata = summary.get("metadata")
    if isinstance(metadata, Mapping) and metadata:
        lines.extend(_render_metadata_table(metadata))

    lines.extend(
        [
            "| Scenario | Runs | Return Codes | Wall Median (s) | Wall Min-Max (s) | "
            "Command Median (s) |",
            "| --- | ---: | --- | ---: | --- | ---: |",
        ]
    )
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
    dask_lines = _render_dask_summary_table(scenarios)
    if dask_lines:
        lines.extend(dask_lines)
    operation_lines = _render_operation_summary_table(scenarios)
    if operation_lines:
        lines.extend(operation_lines)
    command_lines = _render_command_summary_table(scenarios)
    if command_lines:
        lines.extend(command_lines)
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
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write JSON and Markdown benchmark summary artifacts."""
    summary = summarize_benchmark_runs(results)
    if metadata:
        summary["metadata"] = {
            str(key): str(value)
            for key, value in sorted(metadata.items())
            if value is not None and str(value)
        }
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


def _combine_operation_timings(
    operation_timings: Sequence[OperationTimingMetric],
    command_metrics: Sequence[CommandMetric],
) -> tuple[OperationTimingMetric, ...]:
    commands_by_operation: dict[str, list[CommandMetric]] = {}
    for metric in command_metrics:
        if metric.operation is not None:
            commands_by_operation.setdefault(metric.operation, []).append(metric)

    combined = []
    for timing in operation_timings:
        commands = commands_by_operation.get(timing.operation, [])
        command_seconds = sum(metric.duration_seconds for metric in commands)
        combined.append(
            OperationTimingMetric(
                operation=timing.operation,
                elapsed_seconds=timing.elapsed_seconds,
                command_count=len(commands),
                command_seconds=command_seconds,
                operation_minus_command_seconds=timing.elapsed_seconds - command_seconds,
            )
        )
    return tuple(combined)


def _parse_dask_task_groups(text: str) -> tuple[DaskTaskGroupMetric, ...]:
    task_stream_start = text.find('"name":"task_stream"')
    task_stream_text = text[task_stream_start:] if task_stream_start >= 0 else text
    duration_match = re.search(r'\["duration",\[(.*?)\]\]', task_stream_text)
    name_match = re.search(r'\["name",\[(.*?)\]\]', task_stream_text)
    if not duration_match or not name_match:
        return ()

    try:
        durations_ms = json.loads(f"[{duration_match.group(1)}]")
        names = json.loads(f"[{name_match.group(1)}]")
    except json.JSONDecodeError:
        return ()
    if len(durations_ms) != len(names):
        return ()

    grouped: dict[str, list[float]] = {}
    for name, duration_ms in zip(names, durations_ms):
        grouped.setdefault(str(name), []).append(float(duration_ms) / 1000.0)

    metrics = []
    for name, durations in sorted(grouped.items()):
        metrics.append(
            DaskTaskGroupMetric(
                name=name,
                count=len(durations),
                total_seconds=sum(durations),
                median_seconds=statistics.median(durations),
                max_seconds=max(durations),
            )
        )
    return tuple(metrics)


def _summarize_dask_metrics(results: Sequence[BenchmarkRunResult]) -> dict[str, object]:
    summary: dict[str, object] = {}
    if duration_seconds := _present_values(result.dask_duration_seconds for result in results):
        summary["duration_seconds"] = _stats(duration_seconds)
    if compute_seconds := _present_values(result.dask_compute_seconds for result in results):
        summary["compute_seconds"] = _stats(compute_seconds)
    if gaps := _present_values(result.dask_duration_minus_compute_seconds for result in results):
        summary["duration_minus_compute_seconds"] = _stats(gaps)
    if task_counts := _present_values(result.dask_task_count for result in results):
        summary["task_count"] = _stats(task_counts)

    workers = sorted(set(_present_values(result.dask_workers for result in results)))
    if workers:
        summary["workers"] = workers
    threads = sorted(set(_present_values(result.dask_threads for result in results)))
    if threads:
        summary["threads"] = threads
    memory = sorted(
        {
            result.dask_memory
            for result in results
            if result.dask_memory is not None and result.dask_memory
        }
    )
    if memory:
        summary["memory"] = memory

    task_group_summary = _summarize_dask_task_groups(results)
    if task_group_summary:
        summary["task_groups"] = task_group_summary
    return summary


def _summarize_operation_timings(results: Sequence[BenchmarkRunResult]) -> dict[str, object]:
    total_elapsed = [
        sum(timing.elapsed_seconds for timing in result.operation_timings)
        for result in results
        if result.operation_timings
    ]
    total_command = [
        sum(timing.command_seconds for timing in result.operation_timings)
        for result in results
        if result.operation_timings
    ]
    summary: dict[str, object] = {}
    if total_elapsed:
        summary["total_elapsed_seconds"] = _stats(total_elapsed)
    if total_command:
        summary["total_command_seconds"] = _stats(total_command)

    grouped: dict[str, dict[str, list[float]]] = {}
    for result in results:
        for timing in result.operation_timings:
            values = grouped.setdefault(
                timing.operation,
                {
                    "elapsed_seconds": [],
                    "command_seconds": [],
                    "command_count": [],
                    "operation_minus_command_seconds": [],
                },
            )
            values["elapsed_seconds"].append(timing.elapsed_seconds)
            values["command_seconds"].append(timing.command_seconds)
            values["command_count"].append(timing.command_count)
            if timing.operation_minus_command_seconds is not None:
                values["operation_minus_command_seconds"].append(
                    timing.operation_minus_command_seconds
                )

    if grouped:
        summary["by_operation"] = {
            operation: {metric: _stats(values) for metric, values in metrics.items()}
            for operation, metrics in sorted(grouped.items())
        }
    return summary


def _summarize_dask_task_groups(results: Sequence[BenchmarkRunResult]) -> dict[str, object]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for result in results:
        for group in result.dask_task_groups:
            values = grouped.setdefault(
                group.name,
                {
                    "count": [],
                    "total_seconds": [],
                    "median_seconds": [],
                    "max_seconds": [],
                },
            )
            values["count"].append(group.count)
            values["total_seconds"].append(group.total_seconds)
            values["median_seconds"].append(group.median_seconds)
            values["max_seconds"].append(group.max_seconds)

    return {
        name: {metric: _stats(values) for metric, values in metrics.items()}
        for name, metrics in sorted(grouped.items())
    }


def _summarize_command_timings(results: Sequence[BenchmarkRunResult]) -> dict[str, object]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for result in results:
        per_run: dict[str, dict[str, float]] = {}
        for command in result.command_timings:
            values = per_run.setdefault(command.name, {"count": 0.0, "seconds": 0.0})
            values["count"] += 1.0
            values["seconds"] += command.duration_seconds
        for name, values in per_run.items():
            summary_values = grouped.setdefault(name, {"count": [], "seconds": []})
            summary_values["count"].append(values["count"])
            summary_values["seconds"].append(values["seconds"])

    return {
        name: {metric: _stats(values) for metric, values in metrics.items()}
        for name, metrics in sorted(grouped.items())
    }


def _render_dask_summary_table(scenarios: Mapping[str, object]) -> list[str]:
    rows = []
    for scenario_id, data in sorted(scenarios.items()):
        if not isinstance(data, Mapping):
            continue
        dask = data.get("dask")
        if not isinstance(dask, Mapping):
            continue
        duration = dask.get("duration_seconds")
        compute = dask.get("compute_seconds")
        gap = dask.get("duration_minus_compute_seconds")
        task_count = dask.get("task_count")
        if not all(isinstance(item, Mapping) for item in (duration, compute, gap, task_count)):
            continue
        rows.append(
            "| {scenario} | {duration:.3f} | {compute:.3f} | {gap:.3f} | "
            "{tasks:.3f} | {workers} | {threads} |".format(
                scenario=scenario_id,
                duration=duration["median"],
                compute=compute["median"],
                gap=gap["median"],
                tasks=task_count["median"],
                workers=", ".join(map(str, dask.get("workers", []))) or "-",
                threads=", ".join(map(str, dask.get("threads", []))) or "-",
            )
        )
    if not rows:
        return []

    return [
        "## Dask Performance",
        "",
        "| Scenario | Report Median (s) | Compute Median (s) | Gap Median (s) | "
        "Task Count Median | Workers | Threads |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        *rows,
        "",
    ]


def _render_operation_summary_table(scenarios: Mapping[str, object]) -> list[str]:
    rows = []
    for scenario_id, data in sorted(scenarios.items()):
        if not isinstance(data, Mapping):
            continue
        operations = data.get("operations")
        if not isinstance(operations, Mapping):
            continue
        by_operation = operations.get("by_operation")
        if not isinstance(by_operation, Mapping):
            continue
        operation_rows = []
        for operation, metrics in sorted(by_operation.items()):
            if not isinstance(metrics, Mapping):
                continue
            elapsed = metrics.get("elapsed_seconds")
            command = metrics.get("command_seconds")
            gap = metrics.get("operation_minus_command_seconds")
            command_count = metrics.get("command_count")
            if not all(isinstance(item, Mapping) for item in (elapsed, command, gap)):
                continue
            operation_rows.append(
                (
                    float(elapsed["median"]),
                    "| {scenario} | {operation} | {elapsed:.3f} | {command:.3f} | "
                    "{gap:.3f} | {count:.3f} |".format(
                        scenario=scenario_id,
                        operation=operation,
                        elapsed=elapsed["median"],
                        command=command["median"],
                        gap=gap["median"],
                        count=command_count["median"] if isinstance(command_count, Mapping) else 0,
                    ),
                )
            )
        rows.extend(row for _elapsed, row in sorted(operation_rows, reverse=True)[:10])
    if not rows:
        return []

    return [
        "## Operation Timing",
        "",
        "| Scenario | Operation | Elapsed Median (s) | Command Median (s) | "
        "Operation-Command Median (s) | Command Count Median |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        *rows,
        "",
    ]


def _render_command_summary_table(scenarios: Mapping[str, object]) -> list[str]:
    rows = []
    for scenario_id, data in sorted(scenarios.items()):
        if not isinstance(data, Mapping):
            continue
        commands = data.get("commands")
        if not isinstance(commands, Mapping):
            continue
        command_rows = []
        for command_name, metrics in sorted(commands.items()):
            if not isinstance(metrics, Mapping):
                continue
            seconds = metrics.get("seconds")
            count = metrics.get("count")
            if not all(isinstance(item, Mapping) for item in (seconds, count)):
                continue
            command_rows.append(
                (
                    float(seconds["median"]),
                    "| {scenario} | {command} | {seconds:.3f} | {count:.3f} |".format(
                        scenario=scenario_id,
                        command=_markdown_cell(str(command_name)),
                        seconds=seconds["median"],
                        count=count["median"],
                    ),
                )
            )
        rows.extend(row for _seconds, row in sorted(command_rows, reverse=True)[:10])
    if not rows:
        return []

    return [
        "## Command Timing",
        "",
        "| Scenario | Command | Total Median (s) | Count Median |",
        "| --- | --- | ---: | ---: |",
        *rows,
        "",
    ]


def _present_values(values: Iterable[object]) -> list:
    return [value for value in values if value is not None]


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _optional_float_match(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _optional_int_match(pattern: str, text: str) -> Optional[int]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _optional_text_match(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _duration_text_to_seconds(value: str) -> float:
    days = 0
    time_text = value.strip()
    day_match = re.match(r"(?P<days>\d+)\s+days?,\s+(?P<time>.*)", time_text)
    if day_match:
        days = int(day_match.group("days"))
        time_text = day_match.group("time")

    hours_text, minutes_text, seconds_text = time_text.split(":")
    return days * 86400 + int(hours_text) * 3600 + int(minutes_text) * 60 + float(seconds_text)


def _strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)


def _render_metadata_table(metadata: Mapping[str, object]) -> list[str]:
    lines = [
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key, value in sorted(metadata.items()):
        lines.append(
            "| {label} | {value} |".format(
                label=_markdown_cell(METADATA_LABELS.get(str(key), str(key).replace("_", " "))),
                value=_markdown_cell(str(value)),
            )
        )
    lines.append("")
    return lines


def _markdown_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def _optional_str(value: object) -> Optional[str]:
    if value in (None, "", "None"):
        return None
    return str(value)


def _command_name(command: object) -> str:
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        if command:
            return Path(str(command[0])).name
    return "unknown"
