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
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw


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

DIAGNOSTIC_FIELDS = (
    "nsources",
    "theoretical_rms",
    "min_rms_flat_noise",
    "median_rms_flat_noise",
    "dynamic_range_global_flat_noise",
    "min_rms_true_sky",
    "median_rms_true_sky",
    "dynamic_range_global_true_sky",
)
IMAGE_PREVIEW_PRODUCTS = (
    "field-MFS-image-pb-ast",
    "field-MFS-image-pb",
    "field-MFS-residual",
)
VISUAL_COMPARISON_LIMIT = 12


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
    input_snapshot: dict[str, str] = field(default_factory=dict)


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


def _strategy_path_from_parset(parset_path: Path, *, repo_root: Path) -> Path | None:
    parser = _read_parset(parset_path)
    strategy = parser.get("global", "strategy", fallback="").strip().strip("'\"")
    if not strategy or strategy.lower() == "none":
        return None

    strategy_path = Path(strategy).expanduser()
    candidates = [strategy_path] if strategy_path.is_absolute() else []
    candidates.extend(
        [
            parset_path.parent / strategy_path,
            repo_root / strategy_path,
            _repo_root() / strategy_path,
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.is_file():
            return resolved
    return None


def _snapshot_branch_inputs(prepared: PreparedRun, *, run_root: Path) -> None:
    snapshot_dir = run_root / "inputs" / prepared.side
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    parset_snapshot = snapshot_dir / prepared.parset_path.name
    if prepared.parset_path.resolve(strict=False) != parset_snapshot.resolve(strict=False):
        shutil.copy2(prepared.parset_path, parset_snapshot)
    snapshot = {"parset": str(parset_snapshot.relative_to(run_root))}

    strategy_path = _strategy_path_from_parset(prepared.parset_path, repo_root=prepared.repo_root)
    if strategy_path is not None:
        strategy_snapshot = snapshot_dir / strategy_path.name
        if strategy_path.resolve(strict=False) != strategy_snapshot.resolve(strict=False):
            shutil.copy2(strategy_path, strategy_snapshot)
        snapshot["strategy"] = str(strategy_snapshot.relative_to(run_root))

    prepared.input_snapshot = snapshot


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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _diagnostic_files(work_dir: Path) -> dict[str, Path]:
    plots_dir = work_dir / "plots"
    if not plots_dir.is_dir():
        return {}
    return {
        path.relative_to(plots_dir).as_posix(): path
        for path in sorted(plots_dir.glob("image_*/*.image_diagnostics.json"))
    }


def _numeric_or_none(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _compare_image_diagnostics(
    base_work_dir: Path,
    current_work_dir: Path,
    result: saved_equivalence.ComparisonResult,
) -> None:
    """Record compact image diagnostic deltas from generated JSON outputs."""
    base_files = _diagnostic_files(base_work_dir)
    current_files = _diagnostic_files(current_work_dir)
    all_keys = sorted(set(base_files) | set(current_files))
    diagnostics = result.product_statistics.setdefault("diagnostics", [])
    for key in all_keys:
        if key not in base_files:
            result.warnings.append(f"base missing image diagnostics: {key}")
            continue
        if key not in current_files:
            result.warnings.append(f"current missing image diagnostics: {key}")
            continue
        base_data = _load_json(base_files[key])
        current_data = _load_json(current_files[key])
        operation = Path(key).parts[0]
        sector = Path(key).name.removesuffix(".image_diagnostics.json")
        for field_name in DIAGNOSTIC_FIELDS:
            base_value = _numeric_or_none(base_data.get(field_name))
            current_value = _numeric_or_none(current_data.get(field_name))
            if base_value is None or current_value is None:
                if base_value != current_value:
                    result.warnings.append(
                        f"image diagnostic {field_name} differs in presence for {key}"
                    )
                continue
            delta = float(current_value) - float(base_value)
            denominator = abs(float(base_value))
            diagnostics.append(
                {
                    "operation": operation,
                    "sector": sector,
                    "field": field_name,
                    "reference": float(base_value),
                    "current": float(current_value),
                    "delta": delta,
                    "relative_delta": None if denominator == 0 else delta / denominator,
                }
            )
    result.metrics["diagnostics"] = len(base_files)


def _safe_artifact_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower()


def _first_fits_data_hdu(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None and not getattr(hdu.data, "dtype", None).names:
                data = np.asarray(hdu.data, dtype=float)
                data = np.squeeze(data)
                while data.ndim > 2:
                    data = data[0]
                if data.ndim != 2:
                    raise ValueError(f"Cannot render FITS preview for {path}: shape {data.shape}")
                return data
    raise ValueError(f"Cannot render FITS preview for {path}: no image HDU")


def _array_to_preview(data: np.ndarray, *, max_size: int = 480) -> Image.Image:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        normalized = np.zeros(data.shape, dtype=np.uint8)
    else:
        low, high = np.nanpercentile(finite, [1, 99])
        if not math.isfinite(low) or not math.isfinite(high) or low == high:
            low = float(np.nanmin(finite))
            high = float(np.nanmax(finite))
        if low == high:
            normalized = np.full(data.shape, 127, dtype=np.uint8)
        else:
            clipped = np.clip(np.nan_to_num(data, nan=low), low, high)
            normalized = ((clipped - low) / (high - low) * 255).astype(np.uint8)
    image = Image.fromarray(normalized).convert("RGB")
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image


def _fits_to_preview(path: Path) -> Image.Image:
    return _array_to_preview(_first_fits_data_hdu(path))


def _load_png_preview(path: Path, *, max_size: int = 480) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image


def _side_by_side(
    *,
    title: str,
    reference: Image.Image,
    current: Image.Image,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    padding = 16
    label_height = 24
    title_height = 28
    width = reference.width + current.width + padding * 3
    image_height = max(reference.height, current.height)
    height = title_height + label_height + image_height + padding * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 6), title, fill="black")
    left_x = padding
    right_x = padding * 2 + reference.width
    label_y = title_height
    draw.text((left_x, label_y), "master", fill="black")
    draw.text((right_x, label_y), "current", fill="black")
    image_y = title_height + label_height
    canvas.paste(reference, (left_x, image_y + (image_height - reference.height) // 2))
    canvas.paste(current, (right_x, image_y + (image_height - current.height) // 2))
    canvas.save(output_path)


def _image_product_paths(work_dir: Path) -> dict[str, Path]:
    image_root = work_dir / "images"
    if not image_root.is_dir():
        return {}
    paths = {}
    for path in sorted(image_root.glob("image_*/*")):
        if path.is_file() and (path.name.endswith(".fits") or path.name.endswith(".fits.fz")):
            paths[path.relative_to(image_root).as_posix()] = path
    return paths


def _fits_product_stem(path: str) -> str:
    name = Path(path).name
    for suffix in (".fits.fz", ".fits", ".fit"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _select_image_preview_keys(base_paths: dict[str, Path]) -> list[str]:
    selected = []
    for operation in sorted({Path(key).parts[0] for key in base_paths}):
        operation_keys = sorted(key for key in base_paths if Path(key).parts[0] == operation)
        for product in IMAGE_PREVIEW_PRODUCTS:
            match = next(
                (key for key in operation_keys if _fits_product_stem(key) == product), None
            )
            if match:
                selected.append(match)
    return selected[:VISUAL_COMPARISON_LIMIT]


def _normalized_solution_plot_key(path: Path) -> str:
    parts = path.parts
    name = path.name.replace("scalarphase_dir", "fast_phase_dir")
    return "/".join([*parts[:-1], name])


def _solution_plot_paths(work_dir: Path) -> dict[str, Path]:
    pipeline_root = work_dir / "pipelines"
    if not pipeline_root.is_dir():
        return {}
    paths = {}
    for path in sorted(pipeline_root.glob("calibrate_*/*.png")):
        if path.is_file():
            relative = path.relative_to(pipeline_root)
            paths[_normalized_solution_plot_key(relative)] = path
    return paths


def _select_solution_plot_keys(keys: set[str]) -> list[str]:
    selected = []
    seen_groups = set()
    for key in sorted(keys):
        path = Path(key)
        operation = path.parts[0]
        name = path.name
        if "fast_phase_dir" in name:
            kind = "fast_phase"
        elif "medium1_phase_dir" in name:
            kind = "medium1_phase"
        else:
            continue
        group = (operation, kind)
        if group in seen_groups:
            continue
        seen_groups.add(group)
        selected.append(key)
    return selected[:VISUAL_COMPARISON_LIMIT]


def _collect_visual_comparisons(
    base_work_dir: Path,
    current_work_dir: Path,
    run_dir: Path,
    result: saved_equivalence.ComparisonResult,
) -> None:
    comparisons = result.product_statistics.setdefault("visual_comparisons", [])
    visual_dir = run_dir / "visual-comparisons"

    base_images = _image_product_paths(base_work_dir)
    current_images = _image_product_paths(current_work_dir)
    for key in _select_image_preview_keys(base_images):
        if key not in current_images:
            continue
        title = key
        output = visual_dir / "images" / f"{_safe_artifact_name(key)}.png"
        try:
            _side_by_side(
                title=title,
                reference=_fits_to_preview(base_images[key]),
                current=_fits_to_preview(current_images[key]),
                output_path=output,
            )
        except Exception as error:  # pragma: no cover - defensive report enhancement
            result.warnings.append(f"could not render image visual comparison for {key}: {error}")
            continue
        comparisons.append(
            {
                "kind": "image",
                "label": title,
                "path": output.relative_to(run_dir).as_posix(),
            }
        )

    base_plots = _solution_plot_paths(base_work_dir)
    current_plots = _solution_plot_paths(current_work_dir)
    for key in _select_solution_plot_keys(set(base_plots) & set(current_plots)):
        output = visual_dir / "solutions" / f"{_safe_artifact_name(key)}.png"
        try:
            _side_by_side(
                title=key,
                reference=_load_png_preview(base_plots[key]),
                current=_load_png_preview(current_plots[key]),
                output_path=output,
            )
        except Exception as error:  # pragma: no cover - defensive report enhancement
            result.warnings.append(
                f"could not render solution visual comparison for {key}: {error}"
            )
            continue
        comparisons.append(
            {
                "kind": "solution",
                "label": key,
                "path": output.relative_to(run_dir).as_posix(),
            }
        )
    result.metrics["visual_comparisons"] = len(comparisons)


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
    _compare_image_diagnostics(base_work_dir, current_work_dir, result)
    _collect_visual_comparisons(base_work_dir, current_work_dir, run_dir, result)
    return result


def _run_as_dict(prepared: PreparedRun) -> dict[str, Any]:
    data = asdict(prepared)
    for key in ("repo_root", "run_dir", "parset_path", "work_dir", "log_path"):
        if data[key] is not None:
            data[key] = str(data[key])
    return data


def _format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.3f}%"


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
        "| Side | Ref | Return Code | Parset | Work Dir | Log | Input Snapshot |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for side in ("base", "current"):
        run = report[side]
        snapshot = run.get("input_snapshot") or {}
        snapshot_text = ", ".join(f"{key}: `{value}`" for key, value in snapshot.items())
        if not snapshot_text:
            snapshot_text = "n/a"
        lines.append(
            "| "
            f"{side} | `{run['ref']}` | {run.get('returncode')} | "
            f"`{run['parset_path']}` | `{run['work_dir']}` | "
            f"`{run.get('log_path')}` | {snapshot_text} |"
        )

    lines.extend(
        [
            "",
            "## Comparison Summary",
            "",
            "| Result | Ops | Records | FITS | Image HDUs | Table HDUs | H5 | Text | Diagnostics | Visuals |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| "
            f"{status} | {metrics.get('operations', 0)} | "
            f"{metrics.get('output_records', 0)} | {metrics.get('fits', 0)} | "
            f"{metrics.get('fits_image_hdus', 0)} | "
            f"{metrics.get('fits_table_hdus', 0)} | {metrics.get('h5', 0)} | "
            f"{metrics.get('text', 0)} | {metrics.get('diagnostics', 0)} | "
            f"{metrics.get('visual_comparisons', 0)} |",
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

    diagnostics = comparison["product_statistics"].get("diagnostics", [])
    if diagnostics:
        lines.extend(
            [
                "",
                "## Image Diagnostics",
                "",
                "| Operation | Sector | Field | Reference | Current | Delta | Relative Delta |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric in diagnostics[:80]:
            lines.append(
                "| "
                f"`{metric['operation']}` | `{metric['sector']}` | `{metric['field']}` | "
                f"{saved_equivalence._format_metric(metric.get('reference'))} | "
                f"{saved_equivalence._format_metric(metric.get('current'))} | "
                f"{saved_equivalence._format_metric(metric.get('delta'))} | "
                f"{_format_percent(metric.get('relative_delta'))} |"
            )
        if len(diagnostics) > 80:
            lines.append(f"| ... {len(diagnostics) - 80} more diagnostic row(s) | | | | | | |")

    visual_comparisons = comparison["product_statistics"].get("visual_comparisons", [])
    if visual_comparisons:
        lines.extend(["", "## Visual Comparisons", ""])
        for item in visual_comparisons[:20]:
            path = item["path"]
            label = item["label"]
            lines.extend(
                [
                    f"### {item['kind'].title()}: `{label}`",
                    "",
                    f"![{label}]({path})",
                    "",
                ]
            )
        if len(visual_comparisons) > 20:
            lines.append(f"... {len(visual_comparisons) - 20} more visual comparison(s)")

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
    for prepared in (base, current):
        _snapshot_branch_inputs(prepared, run_root=run_root)

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
        "input_snapshots": {
            "base": base.input_snapshot,
            "current": current.input_snapshot,
        },
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
