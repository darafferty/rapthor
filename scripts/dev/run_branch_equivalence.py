#!/usr/bin/env python3
"""Run Rapthor from two prepared branch inputs and compare generated products.

This runner intentionally does not translate parsets or strategies. Prepare one
parset/strategy pair for the base ref and one for the current branch, then pass
the parsets with ``--base-parset`` and ``--current-parset``. The script handles
checkout/bootstrap, execution, reporting, and product comparison only.

Because Rapthor is run from each branch checkout, relative paths inside the
parsets are interpreted by that branch's normal runtime. Prefer absolute paths
for shared data and strategy files when comparing across separate checkouts.
"""

from __future__ import annotations

import argparse
import configparser
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


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


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved.resolve(strict=False)


def _read_parset(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    with path.open(encoding="utf-8") as handle:
        parser.read_file(handle)
    return parser


def _work_dir_from_parset(
    parset_path: Path,
    *,
    repo_root: Path,
    override: Path | None = None,
) -> Path:
    if override is not None:
        return _resolve_path(override, base_dir=repo_root)

    parser = _read_parset(parset_path)
    work_dir = parser.get("global", "dir_working", fallback="").strip()
    if not work_dir:
        raise ValueError(f"{parset_path} is missing required global.dir_working")
    return _resolve_path(Path(work_dir), base_dir=repo_root)


def prepare_branch_input(
    *,
    side: str,
    ref: str,
    repo_root: Path,
    parset_path: Path,
    run_dir: Path,
    work_dir_override: Path | None = None,
) -> PreparedRun:
    """Validate one prepared parset and derive the work directory to compare."""
    parset_path = _resolve_existing_path(
        parset_path,
        base_dir=_repo_root(),
        description=f"{side} parset",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return PreparedRun(
        side=side,
        ref=ref,
        repo_root=repo_root,
        run_dir=run_dir,
        parset_path=parset_path,
        work_dir=_work_dir_from_parset(
            parset_path,
            repo_root=repo_root,
            override=work_dir_override,
        ),
    )


def _run_rapthor_from_repo(
    prepared: PreparedRun,
    *,
    runner_command: Sequence[str] | None = None,
) -> PreparedRun:
    env = os.environ.copy()
    env.setdefault("PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS", "180")
    env["PREFECT_HOME"] = str(prepared.run_dir / "prefect-home")
    env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_RAPTHOR"] = "0.0.0"
    if runner_command is None:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(prepared.repo_root), *(item for item in [env.get("PYTHONPATH")] if item)]
        )
        path_prefixes = _rapthor_path_prefixes(prepared.repo_root)
        command = _rapthor_command_for_repo(prepared.repo_root, prepared.parset_path)
    else:
        env.pop("PYTHONPATH", None)
        path_prefixes = _command_path_prefixes(runner_command)
        command = [*runner_command, str(prepared.parset_path)]
    env["PATH"] = os.pathsep.join(
        [
            *(str(path) for path in path_prefixes if path.is_dir()),
            env.get("PATH", ""),
        ]
    )
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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    prepared.command = command
    prepared.returncode = completed.returncode
    prepared.log_path = log_path
    return prepared


def _rapthor_path_prefixes(repo_root: Path) -> tuple[Path, Path]:
    return repo_root / "bin", repo_root / "rapthor" / "scripts"


def _command_path_prefixes(command: Sequence[str]) -> tuple[Path, ...]:
    if not command:
        return ()
    executable = Path(command[0]).expanduser()
    if executable.parent == Path("."):
        return ()
    return (executable.parent.resolve(strict=False),)


def _rapthor_command_for_repo(repo_root: Path, parset_path: Path) -> list[str]:
    if (repo_root / "rapthor" / "cli.py").is_file():
        return [sys.executable, "-m", "rapthor.cli", str(parset_path)]
    legacy_script = repo_root / "bin" / "rapthor"
    if legacy_script.is_file():
        return [sys.executable, str(legacy_script), str(parset_path)]
    return [sys.executable, "-m", "rapthor.cli", str(parset_path)]


def _venv_bin_dir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def _venv_python(venv_dir: Path) -> Path:
    return _venv_bin_dir(venv_dir) / ("python.exe" if os.name == "nt" else "python")


def _venv_rapthor(venv_dir: Path) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return _venv_bin_dir(venv_dir) / f"rapthor{suffix}"


def _git_revision(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _base_env_marker_payload(
    repo_root: Path,
    install_spec: str,
    *,
    system_site_packages: bool,
) -> dict[str, str | bool]:
    return {
        "checkout": str(repo_root.resolve()),
        "revision": _git_revision(repo_root),
        "install_spec": install_spec,
        "system_site_packages": system_site_packages,
    }


def _create_virtual_environment(venv_dir: Path, *, system_site_packages: bool) -> None:
    venv_command = [sys.executable, "-m", "venv"]
    virtualenv_command = [sys.executable, "-m", "virtualenv"]
    if system_site_packages:
        venv_command.append("--system-site-packages")
        virtualenv_command.append("--system-site-packages")
    venv_command.append(str(venv_dir))
    virtualenv_command.append(str(venv_dir))
    try:
        subprocess.run(venv_command, check=True)
    except subprocess.CalledProcessError:
        subprocess.run(virtualenv_command, check=True)


def _virtual_environment_uses_system_site_packages(venv_dir: Path) -> bool:
    config_path = venv_dir / "pyvenv.cfg"
    if not config_path.is_file():
        return False
    for line in config_path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "include-system-site-packages":
            return value.strip().lower() == "true"
    return False


def _virtual_environment_has_pip(venv_dir: Path) -> bool:
    python = _venv_python(venv_dir)
    if not python.is_file():
        return False
    completed = subprocess.run(
        [str(python), "-m", "pip", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.returncode == 0


def _setup_base_python_environment(
    *,
    repo_root: Path,
    venv_dir: Path,
    install_spec: str,
    system_site_packages: bool = False,
    reinstall: bool = False,
) -> Path:
    """Create or refresh a base-branch venv and return its Rapthor executable."""
    venv_dir = venv_dir.resolve()
    needs_create = not _virtual_environment_has_pip(venv_dir)
    needs_create = needs_create or (
        _venv_python(venv_dir).is_file()
        and _virtual_environment_uses_system_site_packages(venv_dir) != system_site_packages
    )
    if needs_create:
        _create_virtual_environment(venv_dir, system_site_packages=system_site_packages)
    python = _venv_python(venv_dir)

    marker_path = venv_dir / ".rapthor-branch-equivalence.json"
    desired_marker = _base_env_marker_payload(
        repo_root,
        install_spec,
        system_site_packages=system_site_packages,
    )
    installed_marker = None
    if marker_path.is_file():
        try:
            installed_marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            installed_marker = None

    if reinstall or installed_marker != desired_marker:
        subprocess.run(
            [str(python), "-m", "pip", "install", "-e", install_spec],
            cwd=repo_root,
            check=True,
        )
        marker_path.write_text(json.dumps(desired_marker, indent=2), encoding="utf-8")

    rapthor = _venv_rapthor(venv_dir)
    if not rapthor.is_file():
        raise FileNotFoundError(f"Installed Rapthor executable not found: {rapthor}")
    return rapthor


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
        "| Side | Ref | Return Code | Parset | Work Dir | Log |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for side in ("base", "current"):
        run = report[side]
        lines.append(
            "| "
            f"{side} | `{run['ref']}` | {run.get('returncode')} | "
            f"`{run['parset_path']}` | `{run['work_dir']}` | "
            f"`{run.get('log_path')}` |"
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

    if comparison["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in comparison["warnings"][:50]:
            lines.append(f"- {warning}")
        if len(comparison["warnings"]) > 50:
            lines.append(f"- ... {len(comparison['warnings']) - 50} more warning(s)")

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
    comparison: saved_equivalence.ComparisonResult,
) -> None:
    report = {
        "scenario_id": scenario_id,
        "run_root": str(run_root),
        "base": _run_as_dict(base),
        "current": _run_as_dict(current),
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
        "base_work_dir": str(base.work_dir),
        "current_work_dir": str(current.work_dir),
    }
    (run_root / "scenario-manifest.json").write_text(
        json.dumps(scenario_manifest, indent=2),
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
        "--base-parset",
        type=Path,
        required=True,
        help=(
            "Prepared parset for the base ref. It is run as supplied and must "
            "point at the base-compatible strategy and input data."
        ),
    )
    parser.add_argument(
        "--current-parset",
        type=Path,
        required=True,
        help=(
            "Prepared parset for the current branch. It is run as supplied and "
            "must point at the current-branch strategy and input data."
        ),
    )
    parser.add_argument("--scenario-id", help="Label for reports. Defaults to the parset stems.")
    parser.add_argument("--base-ref", default="master", help="Base git ref to compare against.")
    parser.add_argument(
        "--base-checkout",
        type=Path,
        help="Existing checkout for the base ref. Skips git worktree creation.",
    )
    parser.add_argument(
        "--setup-base-env",
        action="store_true",
        help=(
            "Create or reuse a virtualenv for the base checkout and run the base "
            "side with that environment's installed rapthor executable."
        ),
    )
    parser.add_argument(
        "--base-venv",
        type=Path,
        help="Base virtualenv directory. Defaults to .venv inside the base checkout.",
    )
    parser.add_argument(
        "--base-install-spec",
        default=".",
        help=(
            "Editable pip install target for the base checkout; use '.[dev]' if "
            "the base environment needs development extras."
        ),
    )
    parser.add_argument(
        "--base-system-site-packages",
        action="store_true",
        help=(
            "Create the base virtualenv with access to system site packages. "
            "Useful in prepared containers where compiled astronomy packages "
            "such as python-casacore and everybeam are installed globally."
        ),
    )
    parser.add_argument(
        "--reinstall-base-env",
        action="store_true",
        help="Force reinstalling the base checkout into its virtualenv.",
    )
    parser.add_argument(
        "--current-checkout",
        type=Path,
        help="Existing checkout for the current branch. Defaults to this repository.",
    )
    parser.add_argument(
        "--base-work-dir",
        type=Path,
        help=(
            "Work directory to compare for the base run. Defaults to "
            "global.dir_working from --base-parset, resolved from the base checkout."
        ),
    )
    parser.add_argument(
        "--current-work-dir",
        type=Path,
        help=(
            "Work directory to compare for the current run. Defaults to "
            "global.dir_working from --current-parset, resolved from the current checkout."
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Directory for command logs, manifests, and comparison reports.",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate inputs and write manifests/reports without running Rapthor.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Write reports but exit successfully when runs or comparison fail.",
    )
    return parser.parse_args(argv)


def _default_scenario_id(base_parset: Path, current_parset: Path) -> str:
    if base_parset.stem == current_parset.stem:
        return base_parset.stem
    return f"{base_parset.stem}-vs-{current_parset.stem}"


def run(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    base_parset = _resolve_existing_path(
        args.base_parset,
        base_dir=repo_root,
        description="Base parset",
    )
    current_parset = _resolve_existing_path(
        args.current_parset,
        base_dir=repo_root,
        description="Current parset",
    )
    scenario_id = args.scenario_id or _default_scenario_id(base_parset, current_parset)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = (args.run_root or Path("runs") / f"branch-equivalence-{stamp}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    needs_base_checkout = bool(args.setup_base_env) or not args.prepare_only
    if args.base_checkout:
        base_repo = _resolve_existing_path(
            args.base_checkout,
            base_dir=repo_root,
            description="Base checkout",
        )
    elif needs_base_checkout:
        base_repo = _git_worktree_checkout(
            repo_root,
            ref=args.base_ref,
            checkout_dir=run_root / "checkouts" / "base",
        )
    else:
        base_repo = repo_root

    base_runner_command = None
    if args.setup_base_env:
        base_venv = (
            (args.base_venv if args.base_venv.is_absolute() else repo_root / args.base_venv)
            if args.base_venv
            else base_repo / ".venv"
        )
        base_rapthor = _setup_base_python_environment(
            repo_root=base_repo,
            venv_dir=base_venv,
            install_spec=args.base_install_spec,
            system_site_packages=args.base_system_site_packages,
            reinstall=args.reinstall_base_env,
        )
        base_runner_command = [str(base_rapthor)]

    current_repo = (
        _resolve_existing_path(
            args.current_checkout,
            base_dir=repo_root,
            description="Current checkout",
        )
        if args.current_checkout
        else repo_root
    )

    base = prepare_branch_input(
        side="base",
        ref=args.base_ref,
        repo_root=base_repo,
        parset_path=base_parset,
        run_dir=run_root / "base",
        work_dir_override=args.base_work_dir,
    )
    if base_runner_command is not None:
        base.command = [*base_runner_command, str(base.parset_path)]

    current = prepare_branch_input(
        side="current",
        ref="current",
        repo_root=current_repo,
        parset_path=current_parset,
        run_dir=run_root / "current",
        work_dir_override=args.current_work_dir,
    )

    if args.prepare_only:
        comparison = saved_equivalence.ComparisonResult(scenario_id, run_root)
        _write_reports(
            run_root=run_root,
            scenario_id=scenario_id,
            base=base,
            current=current,
            comparison=comparison,
        )
        print(f"Prepared branch equivalence run metadata under {run_root}", flush=True)
        return 0

    print(f"=== base: {args.base_ref} ===", flush=True)
    base = _run_rapthor_from_repo(base, runner_command=base_runner_command)
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
        comparison=comparison,
    )
    print(f"Report: {run_root / 'branch-equivalence-report.json'}", flush=True)
    print(f"Markdown report: {run_root / 'branch-equivalence-report.md'}", flush=True)
    return 0 if comparison.passed or args.allow_failures else 1


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
