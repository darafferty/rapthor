#!/usr/bin/env python3
"""Run the Rapthor benchmark baseline scenarios and write report artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from rapthor.execution.benchmarking import (
    benchmark_scenarios_by_id,
    default_benchmark_scenarios,
    run_scenario_once,
    write_summary_artifacts,
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
            "Run committed Rapthor benchmark scenarios. Run from the prepared dev "
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
        default=3,
        help="Number of clean repetitions per scenario. Defaults to 3.",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
