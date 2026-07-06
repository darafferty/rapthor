#!/usr/bin/env python3
"""Run a small matrix of prepared branch-equivalence scenarios.

The branch equivalence runner intentionally compares one explicitly prepared
base/current parset pair at a time. This wrapper keeps option-focused matrices
easy to repeat without moving comparison logic out of
``run_branch_equivalence.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

BRANCH_EQUIVALENCE_SCRIPT = Path(__file__).with_name("run_branch_equivalence.py")


def _load_matrix(path: Path) -> dict[str, Any]:
    matrix = json.loads(path.read_text(encoding="utf-8"))
    scenarios = matrix.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("matrix must contain a non-empty 'scenarios' list")

    seen: set[str] = set()
    for index, scenario in enumerate(scenarios, start=1):
        if not isinstance(scenario, dict):
            raise ValueError(f"scenario {index} must be an object")
        scenario_id = _scenario_id(scenario, index)
        if scenario_id in seen:
            raise ValueError(f"duplicate scenario id: {scenario_id}")
        seen.add(scenario_id)
        if _skip_reason(scenario):
            continue
        for key in ("base_parset", "current_parset"):
            if key not in scenario:
                raise ValueError(f"scenario {scenario_id!r} is missing required '{key}'")
    return matrix


def _scenario_id(scenario: dict[str, Any], index: int) -> str:
    return str(scenario.get("id") or f"scenario-{index:02d}")


def _skip_reason(scenario: dict[str, Any]) -> str | None:
    reason = scenario.get("skip_reason")
    if reason:
        return str(reason)
    if scenario.get("skip"):
        return "skipped by matrix"
    return None


def _resolve_matrix_path(value: str | Path, *, matrix_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = matrix_dir / path
    return path.resolve(strict=False)


def _add_optional_path(
    command: list[str],
    flag: str,
    value: Path | None,
    *,
    base_dir: Path,
) -> None:
    if value is None:
        return
    command.extend([flag, str(_resolve_matrix_path(value, matrix_dir=base_dir))])


def _scenario_repeatability_repetitions(
    scenario: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    repetitions = scenario.get("repeatability_repetitions", args.repeatability_repetitions)
    return int(repetitions or 1)


def _branch_command(
    *,
    scenario: dict[str, Any],
    scenario_id: str,
    scenario_run_root: Path,
    matrix_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    repetitions = _scenario_repeatability_repetitions(scenario, args)
    command = [
        sys.executable,
        str(BRANCH_EQUIVALENCE_SCRIPT),
        "--scenario-id",
        scenario_id,
        "--base-ref",
        str(scenario.get("base_ref", args.base_ref)),
        "--base-parset",
        str(_resolve_matrix_path(scenario["base_parset"], matrix_dir=matrix_dir)),
        "--current-parset",
        str(_resolve_matrix_path(scenario["current_parset"], matrix_dir=matrix_dir)),
        "--run-root",
        str(scenario_run_root),
        "--atol",
        str(scenario.get("atol", args.atol)),
        "--rtol",
        str(scenario.get("rtol", args.rtol)),
    ]

    if repetitions > 1:
        command.extend(["--repeatability-repetitions", str(repetitions)])
        scenario_work_root = scenario.get("repeatability_work_root")
        if scenario_work_root:
            work_root = _resolve_matrix_path(scenario_work_root, matrix_dir=matrix_dir)
        elif args.repeatability_work_root:
            work_root = _resolve_matrix_path(args.repeatability_work_root, matrix_dir=Path.cwd())
            work_root = work_root / scenario_id
        else:
            work_root = None
        if work_root is not None:
            command.extend(["--repeatability-work-root", str(work_root)])

    cli_base_dir = Path.cwd()
    _add_optional_path(command, "--base-checkout", args.base_checkout, base_dir=cli_base_dir)
    _add_optional_path(command, "--base-venv", args.base_venv, base_dir=cli_base_dir)
    _add_optional_path(command, "--current-checkout", args.current_checkout, base_dir=cli_base_dir)

    if args.setup_base_env:
        command.append("--setup-base-env")
    if args.base_install_spec:
        command.extend(["--base-install-spec", args.base_install_spec])
    if args.base_system_site_packages:
        command.append("--base-system-site-packages")
    if args.reinstall_base_env:
        command.append("--reinstall-base-env")
    if args.prepare_only:
        command.append("--prepare-only")
    if args.allow_failures:
        command.append("--allow-failures")
    return command


def _single_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    comparison = report.get("comparison", {})
    return {
        "result": "pass" if comparison.get("passed") else "fail",
        "report": "branch-equivalence-report.json",
        "pairs": 1,
        "passed_pairs": 1 if comparison.get("passed") else 0,
        "failure_count": len(comparison.get("failures", [])),
        "warning_count": len(comparison.get("warnings", [])),
        "metrics": comparison.get("metrics", {}),
    }


def _repeatability_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    pair_summaries = report.get("pair_summaries", [])
    if not pair_summaries:
        return {
            "result": "prepared",
            "report": "repeatability-summary.json",
            "pairs": 0,
            "passed_pairs": 0,
            "failure_count": 0,
            "warning_count": 0,
            "metrics": {},
        }
    passed_pairs = sum(1 for row in pair_summaries if row.get("passed"))
    return {
        "result": "pass" if passed_pairs == len(pair_summaries) else "fail",
        "report": "repeatability-summary.json",
        "pairs": len(pair_summaries),
        "passed_pairs": passed_pairs,
        "failure_count": max(row.get("failure_count", 0) for row in pair_summaries),
        "warning_count": max(row.get("warning_count", 0) for row in pair_summaries),
        "metrics": {},
    }


def _read_scenario_summary(scenario_run_root: Path, *, prepare_only: bool) -> dict[str, Any]:
    repeatability_report = scenario_run_root / "repeatability-summary.json"
    if repeatability_report.exists():
        return _repeatability_report_summary(json.loads(repeatability_report.read_text()))

    single_report = scenario_run_root / "branch-equivalence-report.json"
    if single_report.exists():
        summary = _single_report_summary(json.loads(single_report.read_text()))
        if prepare_only:
            summary["result"] = "prepared"
        return summary

    return {
        "result": "missing-report",
        "report": None,
        "pairs": 0,
        "passed_pairs": 0,
        "failure_count": 0,
        "warning_count": 0,
        "metrics": {},
    }


def _render_markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Rapthor Branch Option Matrix",
        "",
        f"Run root: `{report['run_root']}`",
    ]
    description = report.get("description")
    if description:
        lines.extend(["", description])

    lines.extend(
        [
            "",
            "| Scenario | Result | Command RC | Pairs | Passed Pairs | Failures | Warnings | Report | Notes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in report["scenarios"]:
        report_path = row.get("report") or ""
        if report_path:
            report_path = f"`{report_path}`"
        notes = row.get("notes") or row.get("skip_reason") or ""
        lines.append(
            "| "
            f"`{row['id']}` | {row['result']} | {row.get('returncode')} | "
            f"{row.get('pairs', 0)} | {row.get('passed_pairs', 0)} | "
            f"{row.get('failure_count', 0)} | {row.get('warning_count', 0)} | "
            f"{report_path} | {notes} |"
        )
    return "\n".join(lines) + "\n"


def _write_matrix_summary(
    *,
    run_root: Path,
    matrix_path: Path,
    matrix: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    report = {
        "matrix": str(matrix_path),
        "run_root": str(run_root),
        "description": matrix.get("description", ""),
        "scenarios": rows,
    }
    (run_root / "option-matrix-summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (run_root / "option-matrix-summary.md").write_text(
        _render_markdown_summary(report),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    matrix_path = args.matrix.resolve(strict=True)
    matrix_dir = matrix_path.parent
    matrix = _load_matrix(matrix_path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = Path(args.run_root or Path("runs") / f"branch-option-matrix-{stamp}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(matrix["scenarios"], start=1):
        scenario_id = _scenario_id(scenario, index)
        scenario_run_root = run_root / scenario_id
        scenario_run_root.mkdir(parents=True, exist_ok=True)
        skip_reason = _skip_reason(scenario)
        if skip_reason:
            rows.append(
                {
                    "id": scenario_id,
                    "result": "skipped",
                    "returncode": None,
                    "report": None,
                    "pairs": 0,
                    "passed_pairs": 0,
                    "failure_count": 0,
                    "warning_count": 0,
                    "skip_reason": skip_reason,
                    "notes": scenario.get("notes", ""),
                }
            )
            continue

        command = _branch_command(
            scenario=scenario,
            scenario_id=scenario_id,
            scenario_run_root=scenario_run_root,
            matrix_dir=matrix_dir,
            args=args,
        )
        print(f"=== {scenario_id} ===", flush=True)
        completed = subprocess.run(command, check=False)
        row = {
            "id": scenario_id,
            "returncode": completed.returncode,
            "command": command,
            "notes": scenario.get("notes", ""),
            **_read_scenario_summary(scenario_run_root, prepare_only=args.prepare_only),
        }
        rows.append(row)
        _write_matrix_summary(
            run_root=run_root,
            matrix_path=matrix_path,
            matrix=matrix,
            rows=rows,
        )
        if args.stop_on_failure and row["result"] not in {"pass", "prepared"}:
            break

    _write_matrix_summary(
        run_root=run_root,
        matrix_path=matrix_path,
        matrix=matrix,
        rows=rows,
    )
    failed = any(row["result"] not in {"pass", "prepared", "skipped"} for row in rows)
    return 0 if not failed or args.allow_failures else 1


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--base-ref", default="master")
    parser.add_argument("--base-checkout", type=Path)
    parser.add_argument("--setup-base-env", action="store_true")
    parser.add_argument("--base-venv", type=Path)
    parser.add_argument("--base-install-spec", default=".")
    parser.add_argument("--base-system-site-packages", action="store_true")
    parser.add_argument("--reinstall-base-env", action="store_true")
    parser.add_argument("--current-checkout", type=Path)
    parser.add_argument("--repeatability-repetitions", type=int, default=1)
    parser.add_argument("--repeatability-work-root", type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parsed = parser.parse_args(argv)
    if parsed.repeatability_repetitions < 1:
        parser.error("--repeatability-repetitions must be >= 1")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
