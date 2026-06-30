#!/usr/bin/env python3
"""Run current Rapthor outputs against saved CWL reference artifacts.

The active CWL equivalence tests were removed after the migration cutover, but
the saved reference artifacts remain useful as a scientific regression check.
This script rebuilds current pipeline outputs from those reference parsets and
compares saved images, h5parm solutions, sky models, operation markers, and
output-record shapes.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from astropy.io import fits


REFERENCE_ROOT = Path(".pytest_cache/cwl-reference-artifacts")
EQUIVALENCE_INPUTS = Path(".pytest_cache/equivalence-inputs")
RESOURCE_ROOT = Path("tests/resources")
BEAM_TABLE_RTOL = 5e-3
SKIP_REFERENCE_NAMES = {
    "hybrid_screens.failed-missing-idg-20260610122303",
    "normalization.previous-20260610121701",
}
STALE_REFERENCE_NAMES = {
    # Captured before the current diagonal slow-gain contract and explicit
    # post-slow medium-phase strategy. Keep runnable by name, but skip in the
    # default equivalence matrix.
    "dd_slow_gain_calibration",
}

STRATEGY_BY_BASENAME = {
    "dd_fast_medium_strategy.py": EQUIVALENCE_INPUTS / "dd_fast_medium_strategy.py",
    "dd_slow_gains_strategy.py": EQUIVALENCE_INPUTS / "dd_slow_gains_strategy.py",
    "dd_then_di_fast_medium_strategy.py": EQUIVALENCE_INPUTS
    / "dd_then_di_fast_medium_strategy.py",
    "di_fast_phase_strategy.py": EQUIVALENCE_INPUTS / "di_fast_phase_strategy.py",
    "di_full_jones_strategy.py": EQUIVALENCE_INPUTS / "di_full_jones_strategy.py",
    "di_then_dd_fast_medium_strategy.py": EQUIVALENCE_INPUTS
    / "di_then_dd_fast_medium_strategy.py",
}


@dataclass
class ComparisonResult:
    scenario: str
    run_dir: Path
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.failures


def _write_strategy(path: Path, **overrides: Any) -> None:
    """Write a single-step strategy matching the integration fixture defaults."""
    step = {
        "channel_width_hz": 195312.5,
        "do_slowgain_solve": False,
        "do_fulljones_solve": False,
        "peel_outliers": False,
        "peel_bright_sources": False,
        "fast_timestep_sec": 32.0,
        "medium_timestep_sec": 120.0,
        "slow_timestep_sec": 600.0,
        "do_normalize": False,
        "auto_mask": 5.0,
        "auto_mask_nmiter": 2,
        "threshisl": 3.0,
        "threshpix": 5.0,
        "max_nmiter": 12,
        "regroup_model": True,
        "max_distance": None,
        "do_check": False,
        "target_flux": 0.3,
        "max_directions": 4,
        "do_calibrate": True,
        "do_image": True,
        "calibration_strategy": {"dd": ["fast_phase", "medium_phase"]},
    }
    step.update(overrides)
    path.write_text(f"strategy_steps = [{step!r}]\n", encoding="utf-8")


def _set_synthetic_uvw_geometry(ms_path: Path) -> None:
    import casacore.tables as pt

    ref_wavelength_m = 299792458.0 / 134373474.12109375
    with pt.table(str(ms_path), readonly=False, ack=False) as table:
        uvw = table.getcol("UVW")
        ant1 = table.getcol("ANTENNA1")
        ant2 = table.getcol("ANTENNA2")
        times = table.getcol("TIME")

        unique_times = np.unique(times)
        antennas = np.unique(np.concatenate([ant1, ant2]))
        base_radius_lambda = np.linspace(0.0, 2500.0, len(antennas))
        antenna_index = {antenna: index for index, antenna in enumerate(antennas)}
        time_index = {time_value: index for index, time_value in enumerate(unique_times)}

        positions = {}
        for time_value in unique_times:
            t_index = time_index[time_value]
            for antenna in antennas:
                a_index = antenna_index[antenna]
                theta = (2.0 * np.pi * a_index / len(antennas)) + 0.35 * t_index
                radius_lambda = base_radius_lambda[a_index]
                positions[(time_value, antenna)] = np.array(
                    [
                        np.cos(theta) * radius_lambda * ref_wavelength_m,
                        np.sin(theta) * radius_lambda * ref_wavelength_m,
                        0.0,
                    ]
                )

        for row_index in range(len(uvw)):
            first_position = positions[(times[row_index], ant1[row_index])]
            second_position = positions[(times[row_index], ant2[row_index])]
            uvw[row_index] = second_position - first_position

        table.putcol("UVW", uvw)


def _write_normalization_skymodel(output_path: Path) -> None:
    source_model_path = RESOURCE_ROOT / "integration_apparent_sky.txt"
    output_path.write_text(source_model_path.read_text(encoding="utf-8"), encoding="utf-8")
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(" , , Patch_patch_norm_1, 1:37:41.299, 33.09.35.132\n")
        handle.write(
            "snorm0, POINT, Patch_patch_norm_1, 1:37:41.299, 33.09.35.132, "
            "20.0, [-0.8], false, 148240661.621094, 0, 0, 0\n"
        )


def _prepare_normalization_ms(output_root: Path) -> Path:
    import casacore.tables as pt

    output_root.mkdir(parents=True, exist_ok=True)
    ms_path = output_root / "test_ms_for_normalization.ms"
    if ms_path.exists():
        return ms_path

    source_ms = RESOURCE_ROOT / "test.ms"
    shutil.copytree(source_ms, ms_path)
    _set_synthetic_uvw_geometry(ms_path)
    with pt.table(str(ms_path), readonly=False, ack=False) as table:
        data = table.getcol("DATA")
        data[...] = 0.0j
        table.putcol("DATA", data)

    skymodel_path = output_root / "integration_apparent_sky_normalization.txt"
    _write_normalization_skymodel(skymodel_path)
    predicted_ms = output_root / "test_ms_for_normalization_predicted.ms"
    subprocess.run(
        [
            "DP3",
            f"msin={ms_path}",
            "steps=[predict]",
            "predict.usebeammodel=True",
            "predict.beam_interval=120",
            "predict.beammode=array_factor",
            f"predict.sourcedb={skymodel_path}",
            f"msout={predicted_ms}",
        ],
        check=True,
    )

    rng = np.random.default_rng(0)
    with pt.table(str(predicted_ms), readonly=False, ack=False) as table:
        data = table.getcol("DATA")
        noise = (
            rng.normal(scale=0.05, size=data.shape)
            + 1j * rng.normal(scale=0.05, size=data.shape)
        ).astype(data.dtype)
        table.putcol("DATA", data + noise)
    shutil.rmtree(ms_path)
    predicted_ms.rename(ms_path)
    return ms_path


def _strategy_path_for(saved_path: str, scenario: str, input_root: Path) -> Path:
    basename = Path(saved_path).name
    if scenario == "dd_slow_gain_calibration":
        path = input_root / "strategies" / "dd_slow_gain_with_medium2_strategy.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_strategy(
            path,
            do_slowgain_solve=True,
            calibration_strategy={
                "dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"]
            },
        )
        return path
    if candidate := STRATEGY_BY_BASENAME.get(basename):
        return candidate.resolve()

    strategy_dir = input_root / "strategies"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    if basename == "normalization_strategy.py":
        path = strategy_dir / basename
        _write_strategy(path, do_normalize=True)
        return path
    if basename == "peeling_strategy.py":
        path = strategy_dir / basename
        _write_strategy(path, peel_bright_sources=True)
        return path
    if scenario in {"full_stokes_clean_disabled", "image_cube", "restart"}:
        return (EQUIVALENCE_INPUTS / "dd_fast_medium_strategy.py").resolve()

    raise FileNotFoundError(f"No strategy reconstruction for {saved_path!r}")


def _prepare_parset(reference_dir: Path, run_root: Path, input_root: Path) -> Path:
    scenario = reference_dir.name
    config = configparser.ConfigParser()
    config.read(reference_dir / "integration_template.parset")

    scenario_root = run_root / scenario
    work_dir = scenario_root / "work"
    scratch_dir = scenario_root / "scratch"
    work_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    config["global"]["dir_working"] = str(work_dir)
    config["cluster"]["local_scratch_dir"] = str(scratch_dir)
    config["cluster"]["global_scratch_dir"] = str(scratch_dir)
    config["cluster"]["allow_internet_access"] = "False"
    config["cluster"]["prefect_task_runner"] = "sync"
    config["cluster"]["prefect_command_profile"] = "auto"
    config["global"]["strategy"] = str(
        _strategy_path_for(config["global"]["strategy"], scenario, input_root)
    )

    if scenario == "normalization":
        config["global"]["input_ms"] = str(_prepare_normalization_ms(input_root / "normalization"))

    parset_path = scenario_root / f"{scenario}.parset"
    with parset_path.open("w", encoding="utf-8") as handle:
        config.write(handle)
    return parset_path


def _run_rapthor(parset_path: Path, result: ComparisonResult) -> None:
    env = os.environ.copy()
    env.setdefault("PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS", "180")
    env["PREFECT_HOME"] = str(result.run_dir.parent / "prefect-home")
    env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_RAPTHOR"] = "0.0.0"
    completed = subprocess.run(
        ["rapthor", str(parset_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )
    log_path = result.run_dir / "rapthor-command.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        result.failures.append(
            f"rapthor exited with {completed.returncode}; see {log_path.as_posix()}"
        )


def _operation_dirs(work_dir: Path) -> list[str]:
    pipelines = work_dir / "pipelines"
    if not pipelines.exists():
        return []
    return sorted(path.name for path in pipelines.iterdir() if path.is_dir())


def _compare_operation_state(reference_dir: Path, work_dir: Path, result: ComparisonResult) -> None:
    expected_order_path = reference_dir / "operation_order.json"
    if expected_order_path.exists():
        expected_order = json.loads(expected_order_path.read_text(encoding="utf-8"))
        operations = _operation_dirs(work_dir)
        missing = [operation for operation in expected_order if operation not in operations]
        if missing:
            result.failures.append(f"missing operation directories: {missing}")
        for operation in expected_order:
            if not (work_dir / "pipelines" / operation / ".done").is_file():
                result.failures.append(f"missing .done marker for {operation}")
    result.metrics["operations"] = len(_operation_dirs(work_dir))


def _record_basename_summary(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) >= {"basename", "class"}:
            return {"basename": value.get("basename"), "class": value.get("class")}
        if "path" in value and len(value) <= 3:
            return {**{k: v for k, v in value.items() if k != "path"}, "basename": Path(value["path"]).name}
        summary = {key: _record_basename_summary(val) for key, val in sorted(value.items())}
        return {
            key: val
            for key, val in summary.items()
            if val not in (None, [], {})
        }
    if isinstance(value, list):
        summary = [_record_basename_summary(item) for item in value]
        return [item for item in summary if item not in (None, [], {})]
    return value


def _compare_output_records(reference_dir: Path, work_dir: Path, result: ComparisonResult) -> None:
    compared = 0
    for ref_outputs in sorted((reference_dir / "pipelines").glob("*/.outputs.json")):
        operation = ref_outputs.parent.name
        current_outputs = work_dir / "pipelines" / operation / ".outputs.json"
        if not current_outputs.is_file():
            result.failures.append(f"missing output record for {operation}")
            continue
        ref_data = _record_basename_summary(json.loads(ref_outputs.read_text(encoding="utf-8")))
        current_data = _record_basename_summary(json.loads(current_outputs.read_text(encoding="utf-8")))
        if ref_data != current_data:
            result.warnings.append(f"output-record summary differs for {operation}")
        compared += 1
    result.metrics["output_records"] = compared


def _finite_stats(data: np.ndarray) -> dict[str, float]:
    finite = np.asarray(data[np.isfinite(data)], dtype=float)
    if finite.size == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "rms": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "rms": float(np.sqrt(np.mean(finite * finite))),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _close(left: float, right: float, *, atol: float, rtol: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return bool(np.isclose(left, right, atol=atol, rtol=rtol))


def _compare_fits(reference: Path, current: Path, result: ComparisonResult, atol: float, rtol: float) -> None:
    if not current.is_file():
        result.failures.append(f"missing FITS product: {current}")
        return
    with fits.open(reference, memmap=False) as ref_hdul, fits.open(current, memmap=False) as cur_hdul:
        ref_hdu = next((hdu for hdu in ref_hdul if hdu.data is not None), None)
        cur_hdu = next((hdu for hdu in cur_hdul if hdu.data is not None), None)
        if ref_hdu is None or cur_hdu is None:
            if ref_hdu is not cur_hdu:
                result.failures.append(f"FITS data presence differs for {current.name}")
            return
        ref_data = ref_hdu.data
        cur_data = cur_hdu.data
        if ref_data.shape != cur_data.shape:
            result.failures.append(f"FITS shape differs for {current.name}: {ref_data.shape} != {cur_data.shape}")
            return
        if ref_data.dtype.names:
            if ref_data.dtype.names != cur_data.dtype.names:
                result.failures.append(f"FITS table columns differ for {current.name}")
                return
            for column in ref_data.dtype.names:
                ref_column = np.asarray(ref_data[column])
                cur_column = np.asarray(cur_data[column])
                if np.issubdtype(ref_column.dtype, np.number):
                    if not np.allclose(ref_column, cur_column, atol=atol, rtol=rtol, equal_nan=True):
                        result.failures.append(f"FITS table column differs for {current.name}:{column}")
                elif not np.array_equal(ref_column, cur_column):
                    result.failures.append(f"FITS table column differs for {current.name}:{column}")
            return
        ref_stats = _finite_stats(ref_data)
        cur_stats = _finite_stats(cur_data)
        for key in ("count", "mean", "std", "rms", "min", "max"):
            if key == "count":
                if ref_stats[key] != cur_stats[key]:
                    result.failures.append(f"FITS finite count differs for {current.name}")
                continue
            if not _close(ref_stats[key], cur_stats[key], atol=atol, rtol=rtol):
                result.failures.append(
                    f"FITS {key} differs for {current.name}: {ref_stats[key]} != {cur_stats[key]}"
                )


def _compare_h5_dataset(
    product: str,
    name: str,
    reference: h5py.Dataset,
    current: h5py.Dataset,
    result: ComparisonResult,
    atol: float,
    rtol: float,
) -> None:
    ref_value = reference[()]
    cur_value = current[()]
    if getattr(ref_value, "shape", None) != getattr(cur_value, "shape", None):
        result.failures.append(f"HDF5 dataset shape differs for {product}:{name}")
        return
    if np.issubdtype(reference.dtype, np.number):
        if not np.allclose(ref_value, cur_value, atol=atol, rtol=rtol, equal_nan=True):
            diff = np.asarray(cur_value) - np.asarray(ref_value)
            finite = np.asarray(diff[np.isfinite(diff)], dtype=float)
            max_abs = float(np.max(np.abs(finite))) if finite.size else math.nan
            result.failures.append(
                f"HDF5 numeric dataset differs for {product}:{name} (max_abs={max_abs:g})"
            )
    elif not np.array_equal(np.asarray(ref_value), np.asarray(cur_value)):
        result.failures.append(f"HDF5 dataset differs for {product}:{name}")


def _h5_datasets(handle: h5py.File) -> dict[str, h5py.Dataset]:
    datasets = {}

    def collect(name: str, item: Any) -> None:
        if isinstance(item, h5py.Dataset):
            datasets[name] = item

    handle.visititems(collect)
    return datasets


def _compare_h5(reference: Path, current: Path, result: ComparisonResult, atol: float, rtol: float) -> None:
    if not current.is_file():
        result.failures.append(f"missing HDF5 product: {current}")
        return
    with h5py.File(reference, "r") as ref_h5, h5py.File(current, "r") as cur_h5:
        ref_datasets = _h5_datasets(ref_h5)
        cur_datasets = _h5_datasets(cur_h5)
        if set(ref_datasets) != set(cur_datasets):
            result.failures.append(f"HDF5 dataset names differ for {current.name}")
            return
        for name in sorted(ref_datasets):
            _compare_h5_dataset(
                current.name,
                name,
                ref_datasets[name],
                cur_datasets[name],
                result,
                atol,
                rtol,
            )


def _skymodel_summary(path: Path) -> dict[str, Any]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    patches = {
        parts[2].strip()
        for line in lines
        if len(parts := line.split(",")) >= 3 and parts[2].strip()
    }
    return {"lines": len(lines), "patches": len(patches)}


def _beam_table(path: Path) -> np.ndarray:
    return np.asarray(ast.literal_eval(f"[{path.read_text(encoding='utf-8')} ]"), dtype=float)


def _compare_text_product(
    reference: Path,
    current: Path,
    result: ComparisonResult,
    *,
    atol: float,
    rtol: float,
) -> None:
    if not current.is_file():
        result.failures.append(f"missing text product: {current}")
        return
    if reference.name.endswith("_beams.txt"):
        beam_rtol = max(rtol, BEAM_TABLE_RTOL)
        if not np.allclose(_beam_table(reference), _beam_table(current), atol=atol, rtol=beam_rtol):
            result.failures.append(f"beam table differs for {current.name}")
        return
    if reference.suffix == ".txt" and ("sky" in reference.name or "model" in reference.name):
        if _skymodel_summary(reference) != _skymodel_summary(current):
            result.failures.append(f"sky-model summary differs for {current.name}")
    elif reference.read_text(encoding="utf-8", errors="replace") != current.read_text(
        encoding="utf-8", errors="replace"
    ):
        result.failures.append(f"text product differs for {current.name}")


def _compare_saved_products(
    reference_dir: Path,
    work_dir: Path,
    result: ComparisonResult,
    *,
    atol: float,
    rtol: float,
) -> None:
    counts = {"fits": 0, "h5": 0, "text": 0}
    for category in ("images", "solutions", "skymodels", "regions"):
        reference_category = reference_dir / category
        current_category = work_dir / category
        if not reference_category.exists():
            continue
        for reference in sorted(path for path in reference_category.glob("**/*") if path.is_file()):
            relative = reference.relative_to(reference_category)
            current = current_category / relative
            suffix = reference.suffix.lower()
            if suffix in {".fits", ".fit", ".fz"} or reference.name.endswith(".fits.fz"):
                _compare_fits(reference, current, result, atol, rtol)
                counts["fits"] += 1
            elif suffix in {".h5", ".h5parm"}:
                _compare_h5(reference, current, result, atol, rtol)
                counts["h5"] += 1
            elif suffix in {".txt", ".reg"}:
                _compare_text_product(reference, current, result, atol=atol, rtol=rtol)
                counts["text"] += 1
    result.metrics.update(counts)


def _reference_scenarios(names: list[str], *, include_stale: bool) -> list[Path]:
    skipped = set(SKIP_REFERENCE_NAMES)
    if not names and not include_stale:
        skipped.update(STALE_REFERENCE_NAMES)
    references = [
        path
        for path in sorted(REFERENCE_ROOT.iterdir())
        if path.is_dir() and path.name not in skipped
    ]
    if names:
        selected = {name.strip() for name in names}
        references = [path for path in references if path.name in selected]
        missing = selected - {path.name for path in references}
        if missing:
            raise SystemExit(f"Unknown reference scenario(s): {sorted(missing)}")
    return references


def run(args: argparse.Namespace) -> int:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = Path(args.run_root or tempfile.mkdtemp(prefix=f"rapthor-equivalence-{stamp}-"))
    run_root.mkdir(parents=True, exist_ok=True)
    input_root = run_root / "inputs"

    results: list[ComparisonResult] = []
    for reference_dir in _reference_scenarios(
        args.scenario,
        include_stale=args.include_stale_references,
    ):
        scenario_root = run_root / reference_dir.name
        result = ComparisonResult(reference_dir.name, scenario_root)
        results.append(result)
        print(f"=== {reference_dir.name} ===", flush=True)
        parset_path = _prepare_parset(reference_dir, run_root, input_root)
        _run_rapthor(parset_path, result)
        work_dir = scenario_root / "work"
        if result.failures:
            print(f"FAILED run: {result.failures[-1]}", flush=True)
            if args.stop_on_failure:
                break
            continue
        _compare_operation_state(reference_dir, work_dir, result)
        _compare_output_records(reference_dir, work_dir, result)
        _compare_saved_products(reference_dir, work_dir, result, atol=args.atol, rtol=args.rtol)
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {reference_dir.name}: {result.metrics}", flush=True)
        if result.failures:
            for failure in result.failures[:10]:
                print(f"  - {failure}", flush=True)
            if len(result.failures) > 10:
                print(f"  - ... {len(result.failures) - 10} more failure(s)", flush=True)
            if args.stop_on_failure:
                break

    report = {
        "run_root": str(run_root),
        "results": [
            {
                "scenario": result.scenario,
                "passed": result.passed,
                "run_dir": str(result.run_dir),
                "metrics": result.metrics,
                "failures": result.failures,
                "warnings": result.warnings,
            }
            for result in results
        ],
    }
    report_path = run_root / "equivalence-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}", flush=True)
    return 0 if all(result.passed for result in results) else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="*", help="Reference scenario name(s) to run")
    parser.add_argument("--run-root", help="Directory for generated current-pipeline outputs")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--include-stale-references",
        action="store_true",
        help="Also run references known to encode an older scientific contract.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args(sys.argv[1:])))
