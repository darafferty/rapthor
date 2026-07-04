import importlib.util
import json
import runpy
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "dev" / "run_branch_equivalence.py"


def load_branch_equivalence_script():
    spec = importlib.util.spec_from_file_location("run_branch_equivalence", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_parset(path, strategy_name="strategy.py"):
    path.write_text(
        f"""
[global]
dir_working = old-work
input_ms = data/input.ms
strategy = {strategy_name}

[cluster]
prefect_task_runner = local_dask
local_scratch_dir = old-scratch
global_scratch_dir = old-scratch
""".strip(),
        encoding="utf-8",
    )


def _write_current_strategy(path):
    path.write_text(
        "strategy_steps = [{'do_calibrate': True, 'calibration_strategy': {'di': [], 'dd': ['fast_phase']}}]\n",
        encoding="utf-8",
    )


def test_prepare_branch_inputs_materializes_paths_and_runtime_overrides(tmp_path):
    module = load_branch_equivalence_script()
    parset = tmp_path / "input.parset"
    strategy = tmp_path / "strategy.py"
    _write_parset(parset)
    _write_current_strategy(strategy)

    prepared, changes = module.prepare_branch_inputs(
        side="base",
        ref="master",
        repo_root=tmp_path,
        source_parset=parset,
        run_dir=tmp_path / "run" / "base",
        local_dask_workers=2,
        cpus_per_task=4,
        max_threads=4,
    )

    parser = module._read_parset(prepared.parset_path)
    assert parser.get("global", "input_ms") == str(tmp_path / "data" / "input.ms")
    assert parser.get("global", "dir_working") == str(prepared.work_dir)
    assert parser.get("cluster", "prefect_task_runner") == "sync"
    assert parser.get("cluster", "local_dask_workers") == "2"
    assert Path(parser.get("global", "strategy")).is_file()
    assert any(change.key == "global.input_ms" for change in changes)
    assert any(change.key == "cluster.max_threads" for change in changes)


def test_prepare_branch_inputs_adapts_legacy_strategy_for_current_branch(tmp_path):
    module = load_branch_equivalence_script()
    parset = tmp_path / "input.parset"
    strategy = tmp_path / "strategy.py"
    _write_parset(parset)
    strategy.write_text(
        "strategy_steps = [{'do_calibrate': True, 'do_slowgain_solve': False, 'do_image': False}]\n",
        encoding="utf-8",
    )

    prepared, changes = module.prepare_branch_inputs(
        side="current",
        ref="current",
        repo_root=tmp_path,
        source_parset=parset,
        run_dir=tmp_path / "run" / "current",
    )

    parser = module._read_parset(prepared.parset_path)
    adapted_strategy = Path(parser.get("global", "strategy"))
    steps = runpy.run_path(str(adapted_strategy))["strategy_steps"]
    assert steps[0]["calibration_strategy"] == {
        "di": [],
        "dd": ["fast_phase", "medium_phase"],
    }
    assert "do_slowgain_solve" not in steps[0]
    assert any(change.key == "calibration_strategy" for change in changes)


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


def test_branch_markdown_report_includes_adaptation_summary(tmp_path):
    module = load_branch_equivalence_script()
    report = {
        "scenario_id": "synthetic",
        "run_root": str(tmp_path),
        "base": {
            "ref": "master",
            "returncode": 0,
            "parset_path": "base.parset",
            "work_dir": "base-work",
        },
        "current": {
            "ref": "current",
            "returncode": 0,
            "parset_path": "current.parset",
            "work_dir": "current-work",
        },
        "adaptations": [
            {
                "side": "current",
                "target": "strategy",
                "key": "calibration_strategy",
                "reason": "translated legacy solve toggles",
            }
        ],
        "comparison": {
            "passed": True,
            "metrics": {"operations": 1},
            "product_statistics": {"fits": []},
            "failures": [],
            "warnings": [],
        },
    }

    markdown = module._render_markdown_report(report)

    assert "# Rapthor Branch Equivalence" in markdown
    assert "translated legacy solve toggles" in markdown


def test_prepare_only_writes_reports_without_running_branches(tmp_path):
    module = load_branch_equivalence_script()
    parset = tmp_path / "input.parset"
    strategy = tmp_path / "strategy.py"
    _write_parset(parset)
    _write_current_strategy(strategy)
    run_root = tmp_path / "branch-run"

    args = module.parse_args(
        [
            str(parset),
            "--run-root",
            str(run_root),
            "--prepare-only",
        ]
    )

    assert module.run(args) == 0
    assert (run_root / "branch-equivalence-report.json").is_file()
    assert (run_root / "branch-equivalence-report.md").is_file()
    assert (run_root / "scenario-manifest.json").is_file()
    assert (run_root / "adaptation-manifest.json").is_file()
