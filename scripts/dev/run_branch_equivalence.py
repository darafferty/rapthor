#!/usr/bin/env python3
"""Run Rapthor from two branches and compare generated scientific products."""

from __future__ import annotations

import argparse
import configparser
import copy
import importlib.util
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from rapthor.lib.parset_paths import PARSET_PATH_OPTIONS, resolve_path_value

DEFAULT_PARSET = Path("examples/generated/prefect_demo_rich/prefect_demo_benchmark.parset")
DEFAULT_CURRENT_CALIBRATION_STRATEGY = {
    "dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"],
    "di": [],
}
BUILTIN_STRATEGIES = {"selfcal", "image"}
LEGACY_STRATEGY_KEYS = {
    "do_fulljones_solve",
    "do_slowgain_solve",
    "slow_timestep_joint_sec",
    "slow_timestep_separate_sec",
}


def _load_saved_equivalence_module():
    module_name = "run_saved_cwl_equivalence"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = Path(__file__).with_name(f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


saved_equivalence = _load_saved_equivalence_module()


@dataclass
class AdaptationChange:
    """A single automatic or manual-override adaptation applied before a run."""

    side: str
    target: str
    key: str
    old: Any
    new: Any
    reason: str


@dataclass
class PreparedRun:
    """Prepared branch run inputs and execution metadata."""

    side: str
    ref: str
    repo_root: Path
    run_dir: Path
    parset_path: Path
    work_dir: Path
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    log_path: Path | None = None


def _repo_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    return Path(__file__).resolve().parents[2]


def _resolve_existing_path(path: Path, *, base_dir: Path, description: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    resolved = resolved.resolve(strict=False)
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def _read_parset(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    with path.open(encoding="utf-8") as handle:
        parser.read_file(handle)
    return parser


def _ensure_section(parser: configparser.ConfigParser, section: str) -> None:
    if not parser.has_section(section):
        parser.add_section(section)


def _set_option(
    parser: configparser.ConfigParser,
    *,
    side: str,
    target: str,
    section: str,
    option: str,
    value: str,
    reason: str,
    changes: list[AdaptationChange],
) -> None:
    _ensure_section(parser, section)
    old = parser.get(section, option, fallback=None)
    parser.set(section, option, value)
    if old != value:
        changes.append(
            AdaptationChange(
                side=side,
                target=target,
                key=f"{section}.{option}",
                old=old,
                new=value,
                reason=reason,
            )
        )


def _materialize_path_options(
    parser: configparser.ConfigParser,
    *,
    side: str,
    base_dir: Path,
    changes: list[AdaptationChange],
) -> None:
    for section, options in PARSET_PATH_OPTIONS.items():
        if not parser.has_section(section):
            continue
        for option in sorted(options):
            if not parser.has_option(section, option):
                continue
            old = parser.get(section, option)
            new = resolve_path_value(old, base_dir)
            if old != new:
                parser.set(section, option, new)
                changes.append(
                    AdaptationChange(
                        side=side,
                        target="parset",
                        key=f"{section}.{option}",
                        old=old,
                        new=new,
                        reason="resolved path relative to the input parset",
                    )
                )


def _strategy_value_is_path(value: str) -> bool:
    stripped = value.strip()
    if not stripped or stripped in BUILTIN_STRATEGIES:
        return False
    return True


def _resolve_strategy_source(
    parser: configparser.ConfigParser,
    *,
    parset_path: Path,
    cli_strategy: Path | None,
    cli_base_dir: Path,
) -> tuple[Path | None, str | None]:
    if cli_strategy is not None:
        return (
            _resolve_existing_path(cli_strategy, base_dir=cli_base_dir, description="Strategy"),
            None,
        )

    value = parser.get("global", "strategy", fallback="").strip()
    if not _strategy_value_is_path(value):
        return None, value or None

    return (
        _resolve_existing_path(Path(value), base_dir=parset_path.parent, description="Strategy"),
        None,
    )


def _load_strategy_steps(path: Path) -> list[dict[str, Any]] | None:
    try:
        namespace = runpy.run_path(str(path), init_globals={"field": None})
    except Exception:
        return None
    steps = namespace.get("strategy_steps")
    if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
        return None
    return steps


def _strategy_needs_current_adaptation(steps: list[dict[str, Any]]) -> bool:
    for step in steps:
        if LEGACY_STRATEGY_KEYS & set(step):
            return True
        if step.get("do_calibrate") and "calibration_strategy" not in step:
            return True
    return False


def _calibration_strategy_from_legacy_step(step: dict[str, Any]) -> dict[str, list[str]]:
    if step.get("do_fulljones_solve"):
        return {"di": ["full_jones"], "dd": []}
    if step.get("do_slowgain_solve", True):
        return copy.deepcopy(DEFAULT_CURRENT_CALIBRATION_STRATEGY)
    return {"di": [], "dd": ["fast_phase", "medium_phase"]}


def _adapt_legacy_strategy_for_current(
    steps: list[dict[str, Any]],
    *,
    changes: list[AdaptationChange],
    strategy_target: str,
) -> list[dict[str, Any]]:
    adapted = copy.deepcopy(steps)
    for index, step in enumerate(adapted):
        target = f"{strategy_target}:strategy_steps[{index}]"
        if "slow_timestep_separate_sec" in step and "slow_timestep_sec" not in step:
            step["slow_timestep_sec"] = step["slow_timestep_separate_sec"]
            changes.append(
                AdaptationChange(
                    side="current",
                    target=target,
                    key="slow_timestep_sec",
                    old=None,
                    new=step["slow_timestep_sec"],
                    reason="renamed legacy slow_timestep_separate_sec",
                )
            )
        for key in sorted(LEGACY_STRATEGY_KEYS):
            if key in step:
                changes.append(
                    AdaptationChange(
                        side="current",
                        target=target,
                        key=key,
                        old=step[key],
                        new=None,
                        reason="removed legacy strategy option for current branch",
                    )
                )
                step.pop(key)
        if step.get("do_calibrate") and "calibration_strategy" not in step:
            new_strategy = _calibration_strategy_from_legacy_step(steps[index])
            step["calibration_strategy"] = new_strategy
            changes.append(
                AdaptationChange(
                    side="current",
                    target=target,
                    key="calibration_strategy",
                    old=None,
                    new=new_strategy,
                    reason="translated legacy solve toggles to current calibration_strategy",
                )
            )
    return adapted


def _write_static_strategy(path: Path, steps: list[dict[str, Any]]) -> None:
    path.write_text(f"strategy_steps = {steps!r}\n", encoding="utf-8")


def _prepare_strategy_copy(
    *,
    side: str,
    source: Path | None,
    builtin_value: str | None,
    explicit_override: bool,
    output_dir: Path,
    changes: list[AdaptationChange],
) -> str | None:
    if source is None:
        return builtin_value

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source.name
    if side == "current" and not explicit_override:
        steps = _load_strategy_steps(source)
        if steps is None:
            shutil.copy2(source, destination)
            changes.append(
                AdaptationChange(
                    side=side,
                    target="strategy",
                    key="manual_review",
                    old=str(source),
                    new=str(destination),
                    reason=(
                        "strategy could not be inspected automatically; copied unchanged, "
                        "use --current-strategy if current-branch compatibility is needed"
                    ),
                )
            )
            return str(destination)
        if _strategy_needs_current_adaptation(steps):
            adapted = _adapt_legacy_strategy_for_current(
                steps,
                changes=changes,
                strategy_target=source.name,
            )
            _write_static_strategy(destination, adapted)
            changes.append(
                AdaptationChange(
                    side=side,
                    target="strategy",
                    key="strategy",
                    old=str(source),
                    new=str(destination),
                    reason="wrote static current-branch strategy from legacy strategy_steps",
                )
            )
            return str(destination)

    shutil.copy2(source, destination)
    changes.append(
        AdaptationChange(
            side=side,
            target="strategy",
            key="strategy",
            old=str(source),
            new=str(destination),
            reason="copied strategy into isolated branch input directory",
        )
    )
    return str(destination)


def prepare_branch_inputs(
    *,
    side: str,
    ref: str,
    repo_root: Path,
    source_parset: Path,
    run_dir: Path,
    cli_strategy: Path | None = None,
    explicit_strategy_override: bool = False,
    cli_base_dir: Path | None = None,
    task_runner: str = "sync",
    command_profile: str = "auto",
    local_dask_workers: int | None = None,
    cpus_per_task: int | None = None,
    max_threads: int | None = None,
) -> tuple[PreparedRun, list[AdaptationChange]]:
    """Write branch-specific parset and strategy inputs for one run side."""
    cli_base = Path.cwd() if cli_base_dir is None else cli_base_dir
    changes: list[AdaptationChange] = []
    parser = _read_parset(source_parset)
    strategy_source, builtin_strategy = _resolve_strategy_source(
        parser,
        parset_path=source_parset,
        cli_strategy=cli_strategy,
        cli_base_dir=cli_base,
    )

    _materialize_path_options(
        parser,
        side=side,
        base_dir=source_parset.parent,
        changes=changes,
    )

    strategy_value = _prepare_strategy_copy(
        side=side,
        source=strategy_source,
        builtin_value=builtin_strategy,
        explicit_override=explicit_strategy_override,
        output_dir=run_dir / "inputs",
        changes=changes,
    )
    if strategy_value is not None:
        _set_option(
            parser,
            side=side,
            target="parset",
            section="global",
            option="strategy",
            value=str(strategy_value),
            reason="selected branch-specific strategy input",
            changes=changes,
        )

    work_dir = run_dir / "rapthor-work"
    scratch_dir = run_dir / "scratch"
    _set_option(
        parser,
        side=side,
        target="parset",
        section="global",
        option="dir_working",
        value=str(work_dir),
        reason="isolate branch run products",
        changes=changes,
    )
    _set_option(
        parser,
        side=side,
        target="parset",
        section="cluster",
        option="local_scratch_dir",
        value=str(scratch_dir),
        reason="isolate branch scratch products",
        changes=changes,
    )
    _set_option(
        parser,
        side=side,
        target="parset",
        section="cluster",
        option="global_scratch_dir",
        value=str(scratch_dir),
        reason="isolate branch scratch products",
        changes=changes,
    )
    if task_runner != "keep":
        _set_option(
            parser,
            side=side,
            target="parset",
            section="cluster",
            option="prefect_task_runner",
            value=task_runner,
            reason="normalize equivalence runtime",
            changes=changes,
        )
    _set_option(
        parser,
        side=side,
        target="parset",
        section="cluster",
        option="prefect_command_profile",
        value=command_profile,
        reason="capture comparable command timing",
        changes=changes,
    )
    _set_option(
        parser,
        side=side,
        target="parset",
        section="cluster",
        option="allow_internet_access",
        value="False",
        reason="keep branch equivalence runs offline/reproducible",
        changes=changes,
    )
    for option, value in (
        ("local_dask_workers", local_dask_workers),
        ("cpus_per_task", cpus_per_task),
        ("max_threads", max_threads),
    ):
        if value is not None:
            _set_option(
                parser,
                side=side,
                target="parset",
                section="cluster",
                option=option,
                value=str(value),
                reason="normalize requested runtime resources",
                changes=changes,
            )

    parset_path = run_dir / "inputs" / f"{source_parset.stem}.{side}.parset"
    parset_path.parent.mkdir(parents=True, exist_ok=True)
    with parset_path.open("w", encoding="utf-8") as handle:
        parser.write(handle)

    return (
        PreparedRun(
            side=side,
            ref=ref,
            repo_root=repo_root,
            run_dir=run_dir,
            parset_path=parset_path,
            work_dir=work_dir,
        ),
        changes,
    )


def _run_rapthor_from_repo(prepared: PreparedRun) -> PreparedRun:
    env = os.environ.copy()
    env.setdefault("PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS", "180")
    env["PREFECT_HOME"] = str(prepared.run_dir / "prefect-home")
    env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_RAPTHOR"] = "0.0.0"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(prepared.repo_root), *(item for item in [env.get("PYTHONPATH")] if item)]
    )
    command = [sys.executable, "-m", "rapthor.cli", str(prepared.parset_path)]
    completed = subprocess.run(
        command,
        cwd=prepared.repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )
    log_path = prepared.run_dir / "rapthor-command.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    prepared.command = command
    prepared.returncode = completed.returncode
    prepared.log_path = log_path
    return prepared


def _git_worktree_checkout(repo_root: Path, *, ref: str, checkout_dir: Path) -> Path:
    if checkout_dir.exists():
        return checkout_dir.resolve()
    checkout_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(checkout_dir), ref],
        cwd=repo_root,
        check=True,
    )
    return checkout_dir.resolve()


def _compare_operation_dirs(
    base_work_dir: Path,
    current_work_dir: Path,
    result: saved_equivalence.ComparisonResult,
) -> None:
    base_operations = saved_equivalence._operation_dirs(base_work_dir)
    current_operations = saved_equivalence._operation_dirs(current_work_dir)
    result.metrics["base_operations"] = len(base_operations)
    result.metrics["current_operations"] = len(current_operations)
    result.metrics["operations"] = len(current_operations)
    if base_operations != current_operations:
        result.failures.append(
            f"operation order differs: base={base_operations}, current={current_operations}"
        )
    for operation in base_operations:
        base_done = base_work_dir / "pipelines" / operation / ".done"
        current_done = current_work_dir / "pipelines" / operation / ".done"
        if not base_done.is_file():
            result.failures.append(f"base missing .done marker for {operation}")
        if not current_done.is_file():
            result.failures.append(f"current missing .done marker for {operation}")


def _compare_output_records(
    base_work_dir: Path,
    current_work_dir: Path,
    result: saved_equivalence.ComparisonResult,
) -> None:
    compared = 0
    for base_outputs in sorted((base_work_dir / "pipelines").glob("*/.outputs.json")):
        operation = base_outputs.parent.name
        current_outputs = current_work_dir / "pipelines" / operation / ".outputs.json"
        if not current_outputs.is_file():
            result.failures.append(f"current missing output record for {operation}")
            continue
        base_data = saved_equivalence._record_basename_summary(
            json.loads(base_outputs.read_text(encoding="utf-8"))
        )
        current_data = saved_equivalence._record_basename_summary(
            json.loads(current_outputs.read_text(encoding="utf-8"))
        )
        if base_data != current_data:
            result.warnings.append(f"output-record summary differs for {operation}")
        compared += 1
    result.metrics["output_records"] = compared


def compare_branch_outputs(
    *,
    scenario_id: str,
    base_work_dir: Path,
    current_work_dir: Path,
    run_dir: Path,
    atol: float,
    rtol: float,
) -> saved_equivalence.ComparisonResult:
    result = saved_equivalence.ComparisonResult(scenario_id, run_dir)
    _compare_operation_dirs(base_work_dir, current_work_dir, result)
    _compare_output_records(base_work_dir, current_work_dir, result)
    saved_equivalence._compare_saved_products(
        base_work_dir,
        current_work_dir,
        result,
        atol=atol,
        rtol=rtol,
    )
    return result


def _run_as_dict(prepared: PreparedRun) -> dict[str, Any]:
    data = asdict(prepared)
    for key in ("repo_root", "run_dir", "parset_path", "work_dir", "log_path"):
        if data[key] is not None:
            data[key] = str(data[key])
    return data


def _render_markdown_report(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    metrics = comparison["metrics"]
    status = "pass" if comparison["passed"] else "fail"
    lines = [
        "# Rapthor Branch Equivalence",
        "",
        f"Scenario: `{report['scenario_id']}`",
        f"Run root: `{report['run_root']}`",
        "",
        "## Branch Runs",
        "",
        "| Side | Ref | Return Code | Parset | Work Dir |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for side in ("base", "current"):
        run = report[side]
        lines.append(
            "| "
            f"{side} | `{run['ref']}` | {run.get('returncode')} | "
            f"`{run['parset_path']}` | `{run['work_dir']}` |"
        )

    lines.extend(
        [
            "",
            "## Comparison Summary",
            "",
            "| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| "
            f"{status} | {metrics.get('operations', 0)} | "
            f"{metrics.get('output_records', 0)} | {metrics.get('fits', 0)} | "
            f"{metrics.get('fits_image_hdus', 0)} | "
            f"{metrics.get('fits_table_hdus', 0)} | {metrics.get('h5', 0)} | "
            f"{metrics.get('text', 0)} |",
            "",
            "## FITS Residual Metrics",
            "",
            "| Product | Max Abs Delta | P99 Abs Delta | Residual RMS | RMS / Ref RMS | "
            "RMS / Ref MAD |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    fits_metrics = [
        metric
        for metric in comparison["product_statistics"].get("fits", [])
        if metric.get("kind") == "image"
    ]
    fits_metrics.sort(key=lambda metric: metric.get("max_abs_delta") or 0.0, reverse=True)
    if not fits_metrics:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")
    for metric in fits_metrics[:25]:
        lines.append(
            "| "
            f"`{metric['product']}` | "
            f"{saved_equivalence._format_metric(metric.get('max_abs_delta'))} | "
            f"{saved_equivalence._format_metric(metric.get('p99_abs_delta'))} | "
            f"{saved_equivalence._format_metric(metric.get('residual_rms'))} | "
            f"{saved_equivalence._format_metric(metric.get('residual_rms_over_reference_rms'))} | "
            f"{saved_equivalence._format_metric(metric.get('residual_rms_over_reference_mad_std'))} |"
        )

    if report["adaptations"]:
        lines.extend(["", "## Adaptations", ""])
        for change in report["adaptations"][:50]:
            lines.append(
                f"- `{change['side']}` `{change['target']}` `{change['key']}`: {change['reason']}"
            )
        if len(report["adaptations"]) > 50:
            lines.append(f"- ... {len(report['adaptations']) - 50} more adaptation(s)")

    if comparison["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in comparison["failures"][:50]:
            lines.append(f"- {failure}")
        if len(comparison["failures"]) > 50:
            lines.append(f"- ... {len(comparison['failures']) - 50} more failure(s)")
    return "\n".join(lines) + "\n"


def _write_reports(
    *,
    run_root: Path,
    scenario_id: str,
    base: PreparedRun,
    current: PreparedRun,
    adaptations: list[AdaptationChange],
    comparison: saved_equivalence.ComparisonResult,
) -> None:
    report = {
        "scenario_id": scenario_id,
        "run_root": str(run_root),
        "base": _run_as_dict(base),
        "current": _run_as_dict(current),
        "adaptations": [asdict(change) for change in adaptations],
        "comparison": {
            "passed": comparison.passed,
            "metrics": comparison.metrics,
            "product_statistics": comparison.product_statistics,
            "failures": comparison.failures,
            "warnings": comparison.warnings,
        },
    }
    (run_root / "branch-equivalence-report.json").write_text(
        json.dumps(report, indent=2, default=saved_equivalence._json_default),
        encoding="utf-8",
    )
    (run_root / "branch-equivalence-report.md").write_text(
        _render_markdown_report(report),
        encoding="utf-8",
    )
    scenario_manifest = {
        "scenario_id": scenario_id,
        "base_ref": base.ref,
        "current_ref": current.ref,
        "base_parset": str(base.parset_path),
        "current_parset": str(current.parset_path),
        "adaptation_manifest": str(run_root / "adaptation-manifest.json"),
    }
    (run_root / "scenario-manifest.json").write_text(
        json.dumps(scenario_manifest, indent=2),
        encoding="utf-8",
    )
    (run_root / "adaptation-manifest.json").write_text(
        json.dumps([asdict(change) for change in adaptations], indent=2),
        encoding="utf-8",
    )


def _prepare_comparison_result(
    scenario_id: str,
    run_root: Path,
    base: PreparedRun,
    current: PreparedRun,
) -> saved_equivalence.ComparisonResult:
    result = saved_equivalence.ComparisonResult(scenario_id, run_root)
    for prepared in (base, current):
        if prepared.returncode not in (0, None):
            result.failures.append(
                f"{prepared.side} run exited with {prepared.returncode}; see {prepared.log_path}"
            )
    return result


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "parset",
        nargs="?",
        type=Path,
        default=DEFAULT_PARSET,
        help=(
            "Parset to run through both branches. Defaults to the generated "
            "benchmark/default-like parset."
        ),
    )
    parser.add_argument("--scenario-id", help="Label for reports. Defaults to the parset stem.")
    parser.add_argument("--base-ref", default="master", help="Base git ref to compare against.")
    parser.add_argument(
        "--base-checkout",
        type=Path,
        help="Existing checkout for the base ref. Skips git worktree creation.",
    )
    parser.add_argument(
        "--current-checkout",
        type=Path,
        help="Existing checkout for the current branch. Defaults to this repository.",
    )
    parser.add_argument("--strategy", type=Path, help="Strategy override used for both branches.")
    parser.add_argument("--base-strategy", type=Path, help="Strategy override for the base run.")
    parser.add_argument(
        "--current-strategy",
        type=Path,
        help="Strategy override for the current-branch run.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Directory for branch run products and reports.",
    )
    parser.add_argument(
        "--task-runner",
        default="sync",
        help="Override cluster.prefect_task_runner for both runs; use 'keep' to preserve parset.",
    )
    parser.add_argument("--command-profile", default="auto")
    parser.add_argument("--local-dask-workers", type=int)
    parser.add_argument("--cpus-per-task", type=int)
    parser.add_argument("--max-threads", type=int)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write adapted parsets/strategies and manifests without running Rapthor.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Write reports but exit successfully when runs or comparison fail.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    source_parset = _resolve_existing_path(args.parset, base_dir=repo_root, description="Parset")
    scenario_id = args.scenario_id or source_parset.stem
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = (args.run_root or Path("runs") / f"branch-equivalence-{stamp}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    base_repo = (
        _resolve_existing_path(args.base_checkout, base_dir=repo_root, description="Base checkout")
        if args.base_checkout
        else repo_root
        if args.prepare_only
        else _git_worktree_checkout(
            repo_root,
            ref=args.base_ref,
            checkout_dir=run_root / "checkouts" / "base",
        )
    )
    current_repo = (
        _resolve_existing_path(
            args.current_checkout,
            base_dir=repo_root,
            description="Current checkout",
        )
        if args.current_checkout
        else repo_root
    )

    base_strategy = args.base_strategy or args.strategy
    current_strategy = args.current_strategy or args.strategy
    base, base_changes = prepare_branch_inputs(
        side="base",
        ref=args.base_ref,
        repo_root=base_repo,
        source_parset=source_parset,
        run_dir=run_root / "base",
        cli_strategy=base_strategy,
        explicit_strategy_override=base_strategy is not None,
        cli_base_dir=repo_root,
        task_runner=args.task_runner,
        command_profile=args.command_profile,
        local_dask_workers=args.local_dask_workers,
        cpus_per_task=args.cpus_per_task,
        max_threads=args.max_threads,
    )
    current, current_changes = prepare_branch_inputs(
        side="current",
        ref="current",
        repo_root=current_repo,
        source_parset=source_parset,
        run_dir=run_root / "current",
        cli_strategy=current_strategy,
        explicit_strategy_override=current_strategy is not None,
        cli_base_dir=repo_root,
        task_runner=args.task_runner,
        command_profile=args.command_profile,
        local_dask_workers=args.local_dask_workers,
        cpus_per_task=args.cpus_per_task,
        max_threads=args.max_threads,
    )
    adaptations = [*base_changes, *current_changes]

    if args.prepare_only:
        comparison = saved_equivalence.ComparisonResult(scenario_id, run_root)
        _write_reports(
            run_root=run_root,
            scenario_id=scenario_id,
            base=base,
            current=current,
            adaptations=adaptations,
            comparison=comparison,
        )
        print(f"Prepared branch equivalence inputs under {run_root}", flush=True)
        return 0

    print(f"=== base: {args.base_ref} ===", flush=True)
    base = _run_rapthor_from_repo(base)
    print(f"base return code: {base.returncode}", flush=True)
    print("=== current ===", flush=True)
    current = _run_rapthor_from_repo(current)
    print(f"current return code: {current.returncode}", flush=True)

    comparison = _prepare_comparison_result(scenario_id, run_root, base, current)
    if not comparison.failures:
        comparison = compare_branch_outputs(
            scenario_id=scenario_id,
            base_work_dir=base.work_dir,
            current_work_dir=current.work_dir,
            run_dir=run_root,
            atol=args.atol,
            rtol=args.rtol,
        )
    _write_reports(
        run_root=run_root,
        scenario_id=scenario_id,
        base=base,
        current=current,
        adaptations=adaptations,
        comparison=comparison,
    )
    print(f"Report: {run_root / 'branch-equivalence-report.json'}", flush=True)
    print(f"Markdown report: {run_root / 'branch-equivalence-report.md'}", flush=True)
    return 0 if comparison.passed or args.allow_failures else 1


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
