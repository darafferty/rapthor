import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "dev" / "run_branch_equivalence.py"
RAPTHOR_LOG = """
2026-07-04 09:47:29,801 - DEBUG - rapthor:predict_di_1 - \x1b[35mTime for operation: 0:00:01.250000\x1b[0m
2026-07-04 09:47:29,801 - DEBUG - rapthor:predict_di_1 - \x1b[35m\x1b[35mTime for operation: 0:00:01.250000\x1b[0m\x1b[0m
2026-07-04 09:48:48,220 - DEBUG - rapthor:image_1 - \x1b[35mTime for operation: 0:01:12.500000\x1b[0m
"""


def load_branch_equivalence_script():
    spec = importlib.util.spec_from_file_location("run_branch_equivalence", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_parset(path, work_dir):
    path.write_text(
        f"""
[global]
dir_working = {work_dir}
input_ms = data/input.ms
strategy = strategy.py

[cluster]
local_scratch_dir = old-local-scratch
global_scratch_dir = old-global-scratch
""".strip(),
        encoding="utf-8",
    )


def test_prepare_branch_input_reads_work_dir_from_prepared_parset(tmp_path):
    module = load_branch_equivalence_script()
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    parset = tmp_path / "inputs" / "base.parset"
    parset.parent.mkdir()
    _write_parset(parset, "relative-work")

    prepared = module.prepare_branch_input(
        side="base",
        ref="master",
        repo_root=repo_root,
        parset_path=parset,
        run_dir=tmp_path / "run" / "base",
    )

    assert prepared.parset_path == parset
    assert prepared.work_dir == repo_root / "relative-work"
    assert prepared.run_dir.is_dir()


def test_prepare_branch_input_accepts_explicit_work_dir_override(tmp_path):
    module = load_branch_equivalence_script()
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    parset = tmp_path / "current.parset"
    _write_parset(parset, "parset-work")

    prepared = module.prepare_branch_input(
        side="current",
        ref="current",
        repo_root=repo_root,
        parset_path=parset,
        run_dir=tmp_path / "run" / "current",
        work_dir_override=tmp_path / "actual-work",
    )

    assert prepared.work_dir == tmp_path / "actual-work"


def test_prepare_branch_input_requires_dir_working(tmp_path):
    module = load_branch_equivalence_script()
    parset = tmp_path / "input.parset"
    parset.write_text("[global]\ninput_ms = data/input.ms\n", encoding="utf-8")

    with pytest.raises(ValueError, match="global.dir_working"):
        module.prepare_branch_input(
            side="base",
            ref="master",
            repo_root=tmp_path,
            parset_path=parset,
            run_dir=tmp_path / "run" / "base",
        )


def test_prepare_repeatability_branch_inputs_write_unique_working_parsets(tmp_path):
    module = load_branch_equivalence_script()
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    strategy = tmp_path / "inputs" / "strategy.py"
    strategy.parent.mkdir()
    strategy.write_text("strategy_steps = []\n", encoding="utf-8")
    parset = strategy.parent / "base.parset"
    _write_parset(parset, "stale-work")

    prepared = module._prepare_repeatability_branch_inputs(
        side="base",
        ref="master",
        repo_root=repo_root,
        parset_path=parset,
        run_root=tmp_path / "repeatability",
        work_root=tmp_path / "repeatability" / "work",
        repetitions=3,
    )

    assert sorted(prepared) == [1, 2, 3]
    for rep_index, run in prepared.items():
        rep_label = f"rep-{rep_index:02d}"
        assert run.parset_path == (
            tmp_path / "repeatability" / "inputs" / "base" / rep_label / "base.parset"
        )
        assert run.work_dir == tmp_path / "repeatability" / "work" / "base" / rep_label
        parser = module._read_parset(run.parset_path)
        assert parser.get("global", "dir_working") == str(run.work_dir)
        assert parser.get("global", "strategy") == str(strategy.resolve())
        assert parser.get("cluster", "local_scratch_dir") == str(run.work_dir / "scratch")
        assert parser.get("cluster", "global_scratch_dir") == str(run.work_dir / "scratch")
        assert run.work_dir.parent.is_dir()


def test_prepare_repeatability_branch_inputs_rehomes_archived_absolute_strategy(tmp_path):
    module = load_branch_equivalence_script()
    repo_root = tmp_path / "checkout"
    repo_root.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    strategy = inputs / "archived_strategy.py"
    strategy.write_text("strategy_steps = []\n", encoding="utf-8")
    parset = inputs / "base.parset"
    _write_parset(parset, "stale-work")
    parser = module._read_parset(parset)
    parser.set("global", "strategy", "/old/run/inputs/archived_strategy.py")
    with parset.open("w", encoding="utf-8") as handle:
        parser.write(handle)

    prepared = module._prepare_repeatability_branch_inputs(
        side="base",
        ref="master",
        repo_root=repo_root,
        parset_path=parset,
        run_root=tmp_path / "repeatability",
        work_root=tmp_path / "repeatability" / "work",
        repetitions=2,
    )

    for run in prepared.values():
        parser = module._read_parset(run.parset_path)
        assert parser.get("global", "strategy") == str(strategy.resolve())


def test_prepare_repeatability_branch_inputs_requires_global_section(tmp_path):
    module = load_branch_equivalence_script()
    parset = tmp_path / "broken.parset"
    parset.write_text("[imaging]\nimage_size = 256\n", encoding="utf-8")

    with pytest.raises(ValueError, match="required \\[global\\]"):
        module._prepare_repeatability_branch_inputs(
            side="base",
            ref="master",
            repo_root=tmp_path,
            parset_path=parset,
            run_root=tmp_path / "repeatability",
            work_root=tmp_path / "repeatability" / "work",
            repetitions=2,
        )


def test_repeatability_pairs_cover_same_branch_and_all_cross_branch_pairs():
    module = load_branch_equivalence_script()

    pairs = module._repeatability_pairs(3)

    assert len(pairs) == 15
    assert [pair.group for pair in pairs].count("base-base") == 3
    assert [pair.group for pair in pairs].count("current-current") == 3
    assert [pair.group for pair in pairs].count("base-current") == 9
    assert pairs[0].pair_id == "base-rep-01_vs_base-rep-02"
    assert pairs[-1].pair_id == "base-rep-03_vs_current-rep-03"


def test_rapthor_command_for_repo_uses_legacy_script_when_cli_module_is_absent(tmp_path):
    module = load_branch_equivalence_script()
    legacy_script = tmp_path / "bin" / "rapthor"
    legacy_script.parent.mkdir()
    legacy_script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    command = module._rapthor_command_for_repo(tmp_path, tmp_path / "input.parset")

    assert command == [sys.executable, str(legacy_script), str(tmp_path / "input.parset")]


def test_rapthor_path_prefixes_include_legacy_script_locations(tmp_path):
    module = load_branch_equivalence_script()

    assert module._rapthor_path_prefixes(tmp_path) == (
        tmp_path / "bin",
        tmp_path / "rapthor" / "scripts",
    )


def test_setup_base_python_environment_creates_venv_and_installs_checkout(tmp_path, monkeypatch):
    module = load_branch_equivalence_script()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    venv = checkout / ".venv"
    calls = []

    class Completed:
        stdout = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            completed = Completed()
            completed.stdout = "abc123\n"
            return completed
        if command[:3] == [sys.executable, "-m", "venv"]:
            bindir = module._venv_bin_dir(venv)
            bindir.mkdir(parents=True)
            module._venv_python(venv).write_text("", encoding="utf-8")
            return Completed()
        if command[1:5] == ["-m", "pip", "install", "-e"]:
            module._venv_rapthor(venv).write_text("", encoding="utf-8")
            return Completed()
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rapthor = module._setup_base_python_environment(
        repo_root=checkout,
        venv_dir=venv,
        install_spec=".",
    )

    assert rapthor == module._venv_rapthor(venv)
    assert json.loads((venv / ".rapthor-branch-equivalence.json").read_text()) == {
        "checkout": str(checkout.resolve()),
        "revision": "abc123",
        "install_spec": ".",
        "system_site_packages": False,
    }
    assert any(call[0][:3] == [sys.executable, "-m", "venv"] for call in calls)
    assert any(call[0][1:5] == ["-m", "pip", "install", "-e"] for call in calls)


def test_setup_base_python_environment_can_use_system_site_packages(tmp_path, monkeypatch):
    module = load_branch_equivalence_script()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    venv = checkout / ".venv"
    calls = []

    class Completed:
        stdout = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            completed = Completed()
            completed.stdout = "abc123\n"
            return completed
        if command[:3] == [sys.executable, "-m", "venv"]:
            bindir = module._venv_bin_dir(venv)
            bindir.mkdir(parents=True)
            module._venv_python(venv).write_text("", encoding="utf-8")
            return Completed()
        if command[1:5] == ["-m", "pip", "install", "-e"]:
            module._venv_rapthor(venv).write_text("", encoding="utf-8")
            return Completed()
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._setup_base_python_environment(
        repo_root=checkout,
        venv_dir=venv,
        install_spec=".",
        system_site_packages=True,
    )

    assert any(
        call[0][:3] == [sys.executable, "-m", "venv"] and "--system-site-packages" in call[0]
        for call in calls
    )
    assert json.loads((venv / ".rapthor-branch-equivalence.json").read_text())[
        "system_site_packages"
    ]


def test_setup_base_python_environment_falls_back_to_virtualenv(tmp_path, monkeypatch):
    module = load_branch_equivalence_script()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    venv = checkout / ".venv"
    calls = []

    class Completed:
        stdout = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            completed = Completed()
            completed.stdout = "abc123\n"
            return completed
        if command[:3] == [sys.executable, "-m", "venv"]:
            raise module.subprocess.CalledProcessError(returncode=1, cmd=command)
        if command[:3] == [sys.executable, "-m", "virtualenv"]:
            bindir = module._venv_bin_dir(venv)
            bindir.mkdir(parents=True)
            module._venv_python(venv).write_text("", encoding="utf-8")
            return Completed()
        if command[1:5] == ["-m", "pip", "install", "-e"]:
            module._venv_rapthor(venv).write_text("", encoding="utf-8")
            return Completed()
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rapthor = module._setup_base_python_environment(
        repo_root=checkout,
        venv_dir=venv,
        install_spec=".",
    )

    assert rapthor == module._venv_rapthor(venv)
    assert any(call[0][:3] == [sys.executable, "-m", "venv"] for call in calls)
    assert any(call[0][:3] == [sys.executable, "-m", "virtualenv"] for call in calls)


def test_setup_base_python_environment_repairs_venv_without_pip(tmp_path, monkeypatch):
    module = load_branch_equivalence_script()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    venv = checkout / ".venv"
    module._venv_python(venv).parent.mkdir(parents=True)
    module._venv_python(venv).write_text("", encoding="utf-8")
    calls = []

    class Completed:
        stdout = ""
        returncode = 0

    class Failed:
        stdout = ""
        stderr = ""
        returncode = 1

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            completed = Completed()
            completed.stdout = "abc123\n"
            return completed
        if command == [str(module._venv_python(venv)), "-m", "pip", "--version"]:
            return Failed()
        if command[:3] == [sys.executable, "-m", "venv"]:
            raise module.subprocess.CalledProcessError(returncode=1, cmd=command)
        if command[:3] == [sys.executable, "-m", "virtualenv"]:
            return Completed()
        if command[1:5] == ["-m", "pip", "install", "-e"]:
            module._venv_rapthor(venv).write_text("", encoding="utf-8")
            return Completed()
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rapthor = module._setup_base_python_environment(
        repo_root=checkout,
        venv_dir=venv,
        install_spec=".",
    )

    assert rapthor == module._venv_rapthor(venv)
    assert any(call[0][:3] == [sys.executable, "-m", "virtualenv"] for call in calls)


def test_run_rapthor_with_base_runner_command_uses_isolated_pythonpath(tmp_path, monkeypatch):
    module = load_branch_equivalence_script()
    command_dir = tmp_path / "base" / ".venv" / "bin"
    command_dir.mkdir(parents=True)
    command = command_dir / "rapthor"
    command.write_text("", encoding="utf-8")
    captured = {}

    class Completed:
        returncode = 0
        stdout = "ok\n"

    def fake_run(command_args, **kwargs):
        captured["command"] = command_args
        captured["env"] = kwargs["env"]
        return Completed()

    monkeypatch.setenv("PYTHONPATH", "/current/branch")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    prepared = module.PreparedRun(
        side="base",
        ref="master",
        repo_root=tmp_path / "base",
        run_dir=tmp_path / "run",
        parset_path=tmp_path / "input.parset",
        work_dir=tmp_path / "work",
    )

    result = module._run_rapthor_from_repo(prepared, runner_command=[str(command)])

    assert result.returncode == 0
    assert result.elapsed_seconds == 2.5
    assert captured["command"] == [str(command), str(tmp_path / "input.parset")]
    assert "PYTHONPATH" not in captured["env"]
    assert captured["env"]["PATH"].split(os.pathsep)[0] == str(command_dir)
    assert (tmp_path / "run" / "rapthor-command.log").read_text(encoding="utf-8") == "ok\n"


def test_parse_operation_log_deduplicates_colored_operation_timings(tmp_path):
    module = load_branch_equivalence_script()
    log_path = tmp_path / "rapthor.log"
    log_path.write_text(RAPTHOR_LOG, encoding="utf-8")

    timings = module.parse_operation_log(log_path)

    assert [(timing.operation, timing.elapsed_seconds) for timing in timings] == [
        ("predict_di_1", 1.25),
        ("image_1", 72.5),
    ]


def test_run_rapthor_collects_operation_timings_from_work_log(tmp_path, monkeypatch):
    module = load_branch_equivalence_script()
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    command = command_dir / "rapthor"
    command.write_text("", encoding="utf-8")

    class Completed:
        returncode = 0
        stdout = "stdout only\n"

    def fake_run(*_args, **_kwargs):
        work_log = tmp_path / "work" / "logs" / "rapthor.log"
        work_log.parent.mkdir(parents=True)
        work_log.write_text(RAPTHOR_LOG, encoding="utf-8")
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    prepared = module.PreparedRun(
        side="base",
        ref="master",
        repo_root=tmp_path / "base",
        run_dir=tmp_path / "run",
        parset_path=tmp_path / "input.parset",
        work_dir=tmp_path / "work",
    )

    result = module._run_rapthor_from_repo(prepared, runner_command=[str(command)])

    assert [(timing.operation, timing.elapsed_seconds) for timing in result.operation_timings] == [
        ("predict_di_1", 1.25),
        ("image_1", 72.5),
    ]


def test_runtime_summary_reports_min_median_max_and_delta(tmp_path):
    module = load_branch_equivalence_script()
    base_runs = [
        module.PreparedRun("base", "master", tmp_path, tmp_path, tmp_path, tmp_path),
        module.PreparedRun("base", "master", tmp_path, tmp_path, tmp_path, tmp_path),
        module.PreparedRun("base", "master", tmp_path, tmp_path, tmp_path, tmp_path),
    ]
    current_runs = [
        module.PreparedRun("current", "current", tmp_path, tmp_path, tmp_path, tmp_path),
        module.PreparedRun("current", "current", tmp_path, tmp_path, tmp_path, tmp_path),
        module.PreparedRun("current", "current", tmp_path, tmp_path, tmp_path, tmp_path),
    ]
    for prepared, elapsed in zip(base_runs, [10.0, 12.0, 14.0]):
        prepared.elapsed_seconds = elapsed
    for prepared, elapsed in zip(current_runs, [9.0, 11.0, 13.0]):
        prepared.elapsed_seconds = elapsed

    summary = module._runtime_summary({"base": base_runs, "current": current_runs})

    assert summary["base"] == {
        "count": 3,
        "min_seconds": 10.0,
        "median_seconds": 12.0,
        "max_seconds": 14.0,
    }
    assert summary["current"] == {
        "count": 3,
        "min_seconds": 9.0,
        "median_seconds": 11.0,
        "max_seconds": 13.0,
    }
    assert summary["current_vs_base_median_delta_percent"] == pytest.approx(-1.0 / 12.0)


def test_operation_timing_summary_reports_per_operation_median_delta(tmp_path):
    module = load_branch_equivalence_script()
    base_runs = [
        module.PreparedRun("base", "master", tmp_path, tmp_path, tmp_path, tmp_path),
        module.PreparedRun("base", "master", tmp_path, tmp_path, tmp_path, tmp_path),
    ]
    current_runs = [
        module.PreparedRun("current", "current", tmp_path, tmp_path, tmp_path, tmp_path),
        module.PreparedRun("current", "current", tmp_path, tmp_path, tmp_path, tmp_path),
    ]
    base_runs[0].operation_timings = [
        module.OperationTiming("predict_di_1", 10.0),
        module.OperationTiming("image_1", 100.0),
    ]
    base_runs[1].operation_timings = [
        module.OperationTiming("predict_di_1", 14.0),
        module.OperationTiming("image_1", 120.0),
    ]
    current_runs[0].operation_timings = [
        module.OperationTiming("predict_di_1", 9.0),
        module.OperationTiming("image_1", 80.0),
    ]
    current_runs[1].operation_timings = [
        module.OperationTiming("predict_di_1", 11.0),
        module.OperationTiming("image_1", 100.0),
    ]

    summary = module._operation_timing_summary({"base": base_runs, "current": current_runs})

    assert summary["base"]["operation_count"] == 2
    assert summary["base"]["by_operation"]["predict_di_1"]["median_seconds"] == 12.0
    assert summary["current"]["by_operation"]["image_1"]["median_seconds"] == 90.0
    assert summary["current_vs_base"]["predict_di_1"] == {
        "base_count": 2,
        "base_median_seconds": 12.0,
        "current_count": 2,
        "current_median_seconds": 10.0,
        "current_vs_base_median_delta_percent": pytest.approx(-1.0 / 6.0),
    }


def _write_fits(path, data):
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32)).writeto(path, overwrite=True)


def _write_operation(work_dir, name):
    operation = work_dir / "pipelines" / name
    operation.mkdir(parents=True)
    (operation / ".done").write_text("", encoding="utf-8")
    (operation / ".outputs.json").write_text(
        json.dumps({"image": {"class": "File", "basename": "field-MFS-image.fits"}}),
        encoding="utf-8",
    )


def test_compare_branch_outputs_uses_strengthened_product_checks(tmp_path):
    module = load_branch_equivalence_script()
    base_work = tmp_path / "base-work"
    current_work = tmp_path / "current-work"
    _write_operation(base_work, "image_1")
    _write_operation(current_work, "image_1")
    for work in (base_work, current_work):
        (work / "images").mkdir(parents=True)
        _write_fits(work / "images" / "field-MFS-image.fits", np.ones((4, 4)))

    result = module.compare_branch_outputs(
        scenario_id="synthetic",
        base_work_dir=base_work,
        current_work_dir=current_work,
        run_dir=tmp_path / "run",
        atol=1e-6,
        rtol=1e-3,
    )

    assert result.passed
    assert result.metrics["operations"] == 1
    assert result.metrics["fits_image_hdus"] == 1


def test_compare_branch_outputs_marks_output_record_product_drift_strict(tmp_path):
    module = load_branch_equivalence_script()
    base_work = tmp_path / "base-work"
    current_work = tmp_path / "current-work"
    _write_operation(base_work, "image_1")
    _write_operation(current_work, "image_1")
    (current_work / "pipelines" / "image_1" / ".outputs.json").write_text(
        json.dumps({"image": {"class": "File", "basename": "field-MFS-residual.fits"}}),
        encoding="utf-8",
    )

    result = module.compare_branch_outputs(
        scenario_id="synthetic",
        base_work_dir=base_work,
        current_work_dir=current_work,
        run_dir=tmp_path / "run",
        atol=1e-6,
        rtol=1e-3,
    )

    assert result.failures == ["output-record product basenames differ for image_1"]
    assert result.product_statistics["output_records"][0]["kind"] == "product_basenames"
    classified = {
        (item["category"], item["item"]): item["disposition"]
        for item in result.product_statistics["difference_classification"]
    }
    assert classified[("strict_output_record_products", "image_1")] == "strict-failure"


def test_compare_branch_outputs_warns_for_auxiliary_output_record_drift(tmp_path):
    module = load_branch_equivalence_script()
    base_work = tmp_path / "base-work"
    current_work = tmp_path / "current-work"
    _write_operation(base_work, "calibrate_1")
    _write_operation(current_work, "calibrate_1")
    (base_work / "pipelines" / "calibrate_1" / ".outputs.json").write_text(
        json.dumps({"plot": {"class": "File", "basename": "fast_phase_dir[Patch].png"}}),
        encoding="utf-8",
    )
    (current_work / "pipelines" / "calibrate_1" / ".outputs.json").write_text(
        json.dumps({"plot": {"class": "File", "basename": "scalarphase_dir[Patch].png"}}),
        encoding="utf-8",
    )

    result = module.compare_branch_outputs(
        scenario_id="synthetic",
        base_work_dir=base_work,
        current_work_dir=current_work,
        run_dir=tmp_path / "run",
        atol=1e-6,
        rtol=1e-3,
    )

    assert result.failures == []
    assert result.warnings == ["output-record auxiliary artifact basenames differ for calibrate_1"]
    classified = {
        (item["category"], item["item"]): item["disposition"]
        for item in result.product_statistics["difference_classification"]
    }
    assert classified[("output_record_auxiliary_artifacts", "calibrate_1")] == "warning"


def test_compare_branch_outputs_warns_for_optional_output_record_drift(tmp_path):
    module = load_branch_equivalence_script()
    base_work = tmp_path / "base-work"
    current_work = tmp_path / "current-work"
    _write_operation(base_work, "image_1")
    _write_operation(current_work, "image_1")
    (base_work / "pipelines" / "image_1" / ".outputs.json").write_text(
        json.dumps(
            {
                "image": {"class": "File", "basename": "sector_1-MFS-image-pb.fits"},
                "visibilities": {"class": "Directory", "basename": "test.ms.sector_1.prep"},
            }
        ),
        encoding="utf-8",
    )
    (current_work / "pipelines" / "image_1" / ".outputs.json").write_text(
        json.dumps(
            {
                "image": [
                    {"class": "File", "basename": "sector_1-MFS-image-pb.fits"},
                    {"class": "File", "basename": "sector_1-MFS-image-pb-ast.fits"},
                ],
                "visibilities": {"class": "Directory", "basename": "test.sector_1_prep.ms"},
            }
        ),
        encoding="utf-8",
    )

    result = module.compare_branch_outputs(
        scenario_id="synthetic",
        base_work_dir=base_work,
        current_work_dir=current_work,
        run_dir=tmp_path / "run",
        atol=1e-6,
        rtol=1e-3,
    )

    assert result.failures == []
    assert result.warnings == ["output-record optional artifact basenames differ for image_1"]
    classified = {
        (item["category"], item["item"]): item["disposition"]
        for item in result.product_statistics["difference_classification"]
    }
    assert classified[("output_record_optional_artifacts", "image_1")] == "warning"


def test_compare_branch_outputs_records_image_diagnostics(tmp_path):
    module = load_branch_equivalence_script()
    base_work = tmp_path / "base-work"
    current_work = tmp_path / "current-work"
    _write_operation(base_work, "image_1")
    _write_operation(current_work, "image_1")
    for work, dynamic_range in [(base_work, 100.0), (current_work, 125.0)]:
        diagnostics_dir = work / "plots" / "image_1"
        diagnostics_dir.mkdir(parents=True)
        (diagnostics_dir / "sector_1.image_diagnostics.json").write_text(
            json.dumps(
                {
                    "nsources": 3,
                    "theoretical_rms": 0.1,
                    "min_rms_flat_noise": 0.2,
                    "median_rms_flat_noise": 0.3,
                    "dynamic_range_global_flat_noise": dynamic_range,
                }
            ),
            encoding="utf-8",
        )

    result = module.compare_branch_outputs(
        scenario_id="synthetic",
        base_work_dir=base_work,
        current_work_dir=current_work,
        run_dir=tmp_path / "run",
        atol=1e-6,
        rtol=1e-3,
    )

    diagnostics = result.product_statistics["diagnostics"]
    assert result.metrics["diagnostics"] == 1
    assert {
        "operation": "image_1",
        "sector": "sector_1",
        "field": "dynamic_range_global_flat_noise",
        "reference": 100.0,
        "current": 125.0,
        "delta": 25.0,
        "relative_delta": 0.25,
    } in diagnostics


def test_classify_branch_differences_labels_known_residual_families(tmp_path):
    module = load_branch_equivalence_script()
    result = module.saved_equivalence.ComparisonResult("synthetic", tmp_path / "run")
    result.warnings.append("output-record metadata shape differs for calibrate_1")
    result.warnings.append("output-record auxiliary artifact basenames differ for calibrate_2")
    result.failures.extend(
        [
            "output-record product basenames differ for calibrate_di_1",
            "FITS image pixels differ for field-MFS-image-pb.fits: max_abs_delta=2e-5",
            "FITS image pixels differ for field-MFS-model-pb.fits: max_abs_delta=2e-3",
            "FITS table column differs for sector_1.source_catalog.fits:E_Total_flux",
            "FITS table column differs for sector_1.source_catalog.fits:Total_flux",
            "text product differs for sector_1_facets_ds9.reg",
            "DS9 region geometry differs for sector_2_facets_ds9.reg",
        ]
    )
    result.product_statistics["fits"] = [
        {
            "product": "field-MFS-image-pb.fits",
            "kind": "image",
            "residual_rms_over_reference_rms": 5e-5,
            "p99_abs_delta": 1e-5,
        },
        {
            "product": "field-MFS-model-pb.fits",
            "kind": "image",
            "residual_rms_over_reference_rms": 9e-4,
            "p99_abs_delta": 0.0,
        },
    ]

    module._classify_branch_differences(result)

    classified = {
        (item["category"], item["item"]): item["disposition"]
        for item in result.product_statistics["difference_classification"]
    }
    assert classified[("output_record_metadata_shape", "calibrate_1")] == "warning"
    assert classified[("output_record_auxiliary_artifacts", "calibrate_2")] == "warning"
    assert classified[("strict_output_record_products", "calibrate_di_1")] == "strict-failure"
    assert (
        classified[("small_image_residual", "field-MFS-image-pb.fits")] == "repeatability-candidate"
    )
    assert (
        classified[("sparse_model_image_residual", "field-MFS-model-pb.fits")]
        == "repeatability-candidate"
    )
    assert (
        classified[
            ("pybdsf_catalog_diagnostic_column", "sector_1.source_catalog.fits:E_Total_flux")
        ]
        == "repeatability-candidate"
    )
    assert (
        classified[("strict_fits_table_column", "sector_1.source_catalog.fits:Total_flux")]
        == "strict-failure"
    )
    assert (
        classified[("region_text_formatting", "sector_1_facets_ds9.reg")]
        == "semantic-comparison-needed"
    )
    assert classified[("strict_region_geometry", "sector_2_facets_ds9.reg")] == "strict-failure"
    assert result.metrics["classified_differences"] == 9


def test_compare_branch_outputs_creates_visual_comparisons(tmp_path):
    module = load_branch_equivalence_script()
    base_work = tmp_path / "base-work"
    current_work = tmp_path / "current-work"
    _write_operation(base_work, "image_1")
    _write_operation(current_work, "image_1")
    for work, value in [(base_work, 1.0), (current_work, 1.1)]:
        image_dir = work / "images" / "image_1"
        image_dir.mkdir(parents=True)
        _write_fits(image_dir / "field-MFS-image-pb.fits", np.full((4, 4), value))
        plot_dir = work / "pipelines" / "calibrate_1"
        plot_dir.mkdir(parents=True)
        plot_name = (
            "fast_phase_dir[Patch_0].png" if work == base_work else "scalarphase_dir[Patch_0].png"
        )
        Image.new("RGB", (8, 8), "white").save(plot_dir / plot_name)

    result = module.compare_branch_outputs(
        scenario_id="synthetic",
        base_work_dir=base_work,
        current_work_dir=current_work,
        run_dir=tmp_path / "run",
        atol=1e-6,
        rtol=1e-3,
    )

    visuals = result.product_statistics["visual_comparisons"]
    assert {item["kind"] for item in visuals} == {"image", "solution"}
    assert result.metrics["visual_comparisons"] == 2
    for item in visuals:
        assert (tmp_path / "run" / item["path"]).is_file()


def test_branch_markdown_report_lists_prepared_inputs(tmp_path):
    module = load_branch_equivalence_script()
    report = {
        "scenario_id": "synthetic",
        "run_root": str(tmp_path),
        "base": {
            "ref": "master",
            "returncode": 0,
            "elapsed_seconds": 12.3456,
            "parset_path": "base.parset",
            "work_dir": "base-work",
            "log_path": "base.log",
        },
        "current": {
            "ref": "current",
            "returncode": 0,
            "elapsed_seconds": 10.0,
            "parset_path": "current.parset",
            "work_dir": "current-work",
            "log_path": "current.log",
        },
        "runtime_summary": {
            "base": {
                "count": 1,
                "min_seconds": 12.3456,
                "median_seconds": 12.3456,
                "max_seconds": 12.3456,
            },
            "current": {
                "count": 1,
                "min_seconds": 10.0,
                "median_seconds": 10.0,
                "max_seconds": 10.0,
            },
            "current_vs_base_median_delta_percent": -0.19,
        },
        "operation_timing_summary": {
            "base": {
                "operation_count": 1,
                "by_operation": {
                    "image_1": {
                        "count": 1,
                        "min_seconds": 9.0,
                        "median_seconds": 9.0,
                        "max_seconds": 9.0,
                    }
                },
            },
            "current": {
                "operation_count": 1,
                "by_operation": {
                    "image_1": {
                        "count": 1,
                        "min_seconds": 7.5,
                        "median_seconds": 7.5,
                        "max_seconds": 7.5,
                    }
                },
            },
            "current_vs_base": {
                "image_1": {
                    "base_count": 1,
                    "base_median_seconds": 9.0,
                    "current_count": 1,
                    "current_median_seconds": 7.5,
                    "current_vs_base_median_delta_percent": -1.0 / 6.0,
                }
            },
        },
        "comparison": {
            "passed": True,
            "metrics": {"operations": 1},
            "product_statistics": {
                "fits": [],
                "difference_classification": [
                    {
                        "category": "small_image_residual",
                        "disposition": "repeatability-candidate",
                        "item": "field-MFS-image-pb.fits",
                        "recommendation": "Bound with same-branch scatter.",
                    }
                ],
            },
            "failures": [],
            "warnings": [],
        },
    }

    markdown = module._render_markdown_report(report)

    assert "# Rapthor Branch Equivalence" in markdown
    assert "`base.parset`" in markdown
    assert "`current.parset`" in markdown
    assert "## Runtime Summary" in markdown
    assert "12.346" in markdown
    assert "current-vs-master median delta: -19.000%" in markdown
    assert "## Operation Runtime Summary" in markdown
    assert "| Operation | master Runs | master Median (s) | current Runs |" in markdown
    assert "`image_1` | 1 | 9.000 | 1 | 7.500 | -16.667%" in markdown
    assert "## Difference Classification" in markdown
    assert "`small_image_residual`" in markdown
    assert "## Adaptations" not in markdown


def test_repeatability_pair_markdown_labels_actual_ref_and_repetition(tmp_path):
    module = load_branch_equivalence_script()
    report = {
        "scenario_id": "synthetic:base-rep-01_vs_base-rep-02",
        "run_root": str(tmp_path),
        "repeatability_pair": {
            "pair_id": "base-rep-01_vs_base-rep-02",
            "reference": "base/rep-01",
            "current": "base/rep-02",
        },
        "base": {
            "ref": "master",
            "returncode": 0,
            "elapsed_seconds": 12.0,
            "parset_path": "base-rep-01.parset",
            "work_dir": "base-rep-01-work",
            "log_path": "base-rep-01.log",
        },
        "current": {
            "ref": "master",
            "returncode": 0,
            "elapsed_seconds": 10.0,
            "parset_path": "base-rep-02.parset",
            "work_dir": "base-rep-02-work",
            "log_path": "base-rep-02.log",
        },
        "runtime_summary": {
            "base": {
                "count": 1,
                "min_seconds": 12.0,
                "median_seconds": 12.0,
                "max_seconds": 12.0,
            },
            "current": {
                "count": 1,
                "min_seconds": 10.0,
                "median_seconds": 10.0,
                "max_seconds": 10.0,
            },
            "current_vs_base_median_delta_percent": -1.0 / 6.0,
        },
        "operation_timing_summary": {
            "current_vs_base": {
                "image_1": {
                    "base_count": 1,
                    "base_median_seconds": 9.0,
                    "current_count": 1,
                    "current_median_seconds": 7.5,
                    "current_vs_base_median_delta_percent": -1.0 / 6.0,
                }
            },
        },
        "comparison": {
            "passed": True,
            "metrics": {},
            "product_statistics": {"fits": []},
            "failures": [],
            "warnings": [],
        },
    }

    markdown = module._render_markdown_report(report)

    assert "Repeatability pair: `base-rep-01_vs_base-rep-02`" in markdown
    assert "| master/rep-01 | 1 | 12.000 | 12.000 | 12.000 |" in markdown
    assert "| master/rep-02 | 1 | 10.000 | 10.000 | 10.000 |" in markdown
    assert "master/rep-02-vs-master/rep-01 median delta: -16.667%" in markdown
    assert (
        "| Operation | master/rep-01 Runs | master/rep-01 Median (s) | master/rep-02 Runs |"
        in markdown
    )
    assert "Current-vs-base median delta" not in markdown
    assert "Candidate-vs-reference median delta" not in markdown


def test_prepare_only_writes_reports_without_running_branches(tmp_path):
    module = load_branch_equivalence_script()
    base_parset = tmp_path / "base.parset"
    current_parset = tmp_path / "current.parset"
    _write_parset(base_parset, tmp_path / "base-work")
    _write_parset(current_parset, tmp_path / "current-work")
    (tmp_path / "strategy.py").write_text("strategy_steps = []\n", encoding="utf-8")
    run_root = tmp_path / "branch-run"

    args = module.parse_args(
        [
            "--base-parset",
            str(base_parset),
            "--current-parset",
            str(current_parset),
            "--run-root",
            str(run_root),
            "--prepare-only",
        ]
    )

    assert module.run(args) == 0
    assert (run_root / "branch-equivalence-report.json").is_file()
    assert (run_root / "branch-equivalence-report.md").is_file()
    assert (run_root / "scenario-manifest.json").is_file()
    assert (run_root / "inputs" / "base" / "base.parset").is_file()
    assert (run_root / "inputs" / "base" / "strategy.py").is_file()
    assert (run_root / "inputs" / "current" / "current.parset").is_file()
    assert (run_root / "inputs" / "current" / "strategy.py").is_file()
    assert not (run_root / "adaptation-manifest.json").exists()
    report = json.loads((run_root / "branch-equivalence-report.json").read_text())
    manifest = json.loads((run_root / "scenario-manifest.json").read_text())
    assert "adaptations" not in report
    assert report["base"]["parset_path"] == str(base_parset)
    assert report["runtime_summary"]["base"]["count"] == 0
    assert report["operation_timing_summary"]["current_vs_base"] == {}
    assert report["current"]["parset_path"] == str(current_parset)
    assert report["base"]["input_snapshot"] == {
        "parset": "inputs/base/base.parset",
        "strategy": "inputs/base/strategy.py",
    }
    assert manifest["input_snapshots"]["current"] == {
        "parset": "inputs/current/current.parset",
        "strategy": "inputs/current/strategy.py",
    }


def test_repeatability_prepare_only_writes_summary_and_generated_inputs(tmp_path):
    module = load_branch_equivalence_script()
    base_parset = tmp_path / "base.parset"
    current_parset = tmp_path / "current.parset"
    _write_parset(base_parset, tmp_path / "base-work")
    _write_parset(current_parset, tmp_path / "current-work")
    (tmp_path / "strategy.py").write_text("strategy_steps = []\n", encoding="utf-8")
    run_root = tmp_path / "repeatability-run"

    args = module.parse_args(
        [
            "--base-parset",
            str(base_parset),
            "--current-parset",
            str(current_parset),
            "--run-root",
            str(run_root),
            "--prepare-only",
            "--repeatability-repetitions",
            "2",
        ]
    )

    assert module.run(args) == 0
    summary = json.loads((run_root / "repeatability-summary.json").read_text())
    assert (run_root / "repeatability-summary.md").is_file()
    assert summary["repetitions"] == 2
    assert summary["runtime_summary"]["current"]["count"] == 0
    assert summary["operation_timing_summary"]["current_vs_base"] == {}
    assert len(summary["planned_pairs"]) == 6
    assert summary["pair_summaries"] == []
    assert summary["runs"]["base"]["rep-01"]["work_dir"] == str(
        run_root / "work" / "base" / "rep-01"
    )
    assert (run_root / "inputs" / "current" / "rep-02" / "current.parset").is_file()
