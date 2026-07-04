#!/usr/bin/env python3
"""Run the Rapthor benchmark baseline scenario and write report artifacts."""

from __future__ import annotations

import argparse
import configparser
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from rapthor.execution.benchmark_inputs import ensure_test_ms
from rapthor.execution.benchmarking import (
    benchmark_scenarios_by_id,
    default_benchmark_scenarios,
    format_failed_benchmark_runs,
    run_scenario_once,
    write_summary_artifacts,
)

GENERATED_DEMO_SCENARIO_IDS = frozenset({"ci-benchmark"})
TEST_RESOURCE_DIR = Path("tests/resources")
BENCHMARK_PARSET_PATH_OPTIONS = (
    "input_ms",
    "input_skymodel",
    "apparent_skymodel",
    "strategy",
    "input_h5parm",
    "input_fulljones_h5parm",
    "facet_layout",
)


def _repo_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    return Path(__file__).resolve().parents[2]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    scenario_ids = sorted(benchmark_scenarios_by_id())
    parser = argparse.ArgumentParser(
        description=(
            "Run the Rapthor benchmark scenario. Run from the prepared dev "
            "container when possible so external astronomy tools match CI."
        )
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=scenario_ids,
        help="Scenario to run. Repeat to select more than one. Defaults to all scenarios.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of clean repetitions per scenario. Defaults to 1.",
    )
    parser.add_argument(
        "--local-dask-workers",
        type=int,
        help="Override the local Dask worker count for every selected scenario.",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        help="Override the thread/core count passed to every selected scenario.",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        help="Override cluster.max_threads for external tools in every selected scenario.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/tmp/rapthor-benchmark-baseline"),
        help="Directory for benchmark run products and reports.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="JSON summary artifact. Defaults to <run-root>/benchmark-summary.json.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        help="Markdown report artifact. Defaults to <run-root>/benchmark-report.md.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List committed benchmark scenarios and exit.",
    )
    parser.add_argument(
        "--prepare-inputs",
        action="store_true",
        help=(
            "Download the shared test Measurement Set when needed and generate "
            "ignored benchmark demo inputs before validating scenarios."
        ),
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Write reports but exit successfully even if benchmark repetitions fail.",
    )
    return parser.parse_args(argv)


def _scenario_with_runtime_overrides(scenario, args: argparse.Namespace):
    overrides = {}
    if args.local_dask_workers is not None:
        overrides["local_dask_workers"] = args.local_dask_workers
    if args.cpus_per_task is not None:
        overrides["cpus_per_task"] = args.cpus_per_task
    if args.max_threads is not None:
        overrides["max_threads"] = args.max_threads
    return replace(scenario, **overrides) if overrides else scenario


def _prepare_benchmark_inputs(repo_root: Path, selected_scenarios) -> None:
    """Create ignored benchmark inputs that are not present in a fresh checkout."""
    test_ms = _ensure_test_ms(repo_root)
    generated_scenarios = [
        scenario
        for scenario in selected_scenarios
        if scenario.scenario_id in GENERATED_DEMO_SCENARIO_IDS
    ]
    if generated_scenarios and _missing_scenario_inputs(repo_root, generated_scenarios):
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "dev" / "generate-prefect-demo-data.py"),
                "--template-ms",
                str(test_ms),
                "--force",
            ],
            cwd=repo_root,
            check=True,
        )


def _ensure_test_ms(repo_root: Path) -> Path:
    """Download the shared test Measurement Set when a checkout does not have it."""
    return ensure_test_ms(repo_root / TEST_RESOURCE_DIR)


def _validate_scenario_inputs(repo_root: Path, selected_scenarios) -> None:
    missing = _missing_scenario_inputs(repo_root, selected_scenarios)
    if missing:
        message = "\n".join(["Benchmark inputs are missing:", *[f"- {item}" for item in missing]])
        message += "\nRun again with --prepare-inputs to create ignored benchmark inputs."
        raise SystemExit(message)


def _missing_scenario_inputs(repo_root: Path, selected_scenarios) -> list[str]:
    missing = []
    for scenario in selected_scenarios:
        parset_path = repo_root / scenario.parset
        if not parset_path.exists():
            missing.append(f"{scenario.scenario_id}: parset does not exist: {parset_path}")
            continue

        for option, path_value in _parset_path_values(parset_path):
            if not _path_value_exists(repo_root, path_value):
                missing.append(
                    f"{scenario.scenario_id}: [{parset_path.relative_to(repo_root)}] "
                    f"{option} does not exist: {path_value}"
                )

    return missing


def _parset_path_values(parset_path: Path) -> list[tuple[str, str]]:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(parset_path)
    if not parser.has_section("global"):
        return []

    values = []
    for option in BENCHMARK_PARSET_PATH_OPTIONS:
        if not parser.has_option("global", option):
            continue
        for path_value in _split_parset_path_value(parser.get("global", option)):
            values.append((option, path_value))
    return values


def _split_parset_path_value(value: str) -> list[str]:
    value = value.strip()
    if not value or value.lower() == "none":
        return []
    if value.startswith("[") and value.endswith("]"):
        return [
            item.strip()
            for item in value[1:-1].split(",")
            if item.strip() and item.strip().lower() != "none"
        ]
    return [value]


def _path_value_exists(repo_root: Path, path_value: str) -> bool:
    path = Path(path_value)
    if not path.is_absolute():
        path = repo_root / path
    if any(marker in path.as_posix() for marker in ("*", "?")):
        return bool(list(path.parent.glob(path.name)))
    return path.exists()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    scenarios_by_id = benchmark_scenarios_by_id()
    if args.list_scenarios:
        for scenario in default_benchmark_scenarios():
            print(f"{scenario.scenario_id}: {scenario.description}")
        return 0

    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")

    repo_root = _repo_root()
    run_root = args.run_root.expanduser().resolve()
    output_json = args.output_json or run_root / "benchmark-summary.json"
    output_markdown = args.output_markdown or run_root / "benchmark-report.md"
    selected_ids = args.scenario or sorted(scenarios_by_id)
    selected_scenarios = [
        _scenario_with_runtime_overrides(scenarios_by_id[scenario_id], args)
        for scenario_id in selected_ids
    ]

    if args.prepare_inputs:
        _prepare_benchmark_inputs(repo_root, selected_scenarios)
    _validate_scenario_inputs(repo_root, selected_scenarios)

    results = []
    for scenario in selected_scenarios:
        for repetition in range(1, args.repetitions + 1):
            print(f"Running {scenario.scenario_id} repetition {repetition}/{args.repetitions}")
            results.append(
                run_scenario_once(
                    scenario,
                    repetition=repetition,
                    run_root=run_root,
                    repo_root=repo_root,
                )
            )

    summary = write_summary_artifacts(
        results,
        output_json=output_json,
        output_markdown=output_markdown,
    )
    result_path = run_root / "benchmark-results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote raw results: {result_path}")
    print(f"Wrote JSON summary: {output_json}")
    print(f"Wrote Markdown report: {output_markdown}")
    print(f"Scenarios summarized: {', '.join(sorted(summary['scenarios']))}")
    failures = format_failed_benchmark_runs(results)
    if failures and not args.allow_failures:
        print(failures, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
