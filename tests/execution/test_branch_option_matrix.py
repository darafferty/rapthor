import configparser
import importlib.util
import json
import runpy
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "dev" / "run_branch_option_matrix.py"
OPTION_MATRIX_PATH = Path(__file__).parents[1] / "resources" / "equivalence" / "option-matrix.json"

ACTIVE_EQUIVALENCE_SCENARIOS = {
    "phase-only-core",
    "dd-phase-plus-di-fulljones",
    "normalization-rich-demo",
    "prediction-path-image-based",
    "prediction-path-wsclean",
    "bda-averaging",
    "bda-frequency-limits",
    "initial-skymodel-regroup",
    "initial-skymodel-bda-regroup",
}


def load_branch_option_matrix_script():
    spec = importlib.util.spec_from_file_location("run_branch_option_matrix", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_matrix(path, scenarios):
    path.write_text(
        json.dumps(
            {
                "description": "Option matrix fixture.",
                "scenarios": scenarios,
            }
        ),
        encoding="utf-8",
    )


def test_branch_command_resolves_matrix_relative_inputs(tmp_path):
    module = load_branch_option_matrix_script()
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    for relative in ("base.parset", "current.parset"):
        (matrix_dir / relative).write_text("[global]\ndir_working = work\n", encoding="utf-8")
    scenario = {
        "id": "normalization",
        "base_parset": "base.parset",
        "current_parset": "current.parset",
        "repeatability_repetitions": 3,
    }
    args = module.parse_args(
        [
            "--matrix",
            str(matrix_dir / "matrix.json"),
            "--run-root",
            str(tmp_path / "run"),
            "--repeatability-work-root",
            str(tmp_path / "work"),
            "--setup-base-env",
            "--base-system-site-packages",
            "--base-pip-constraint",
            str(matrix_dir / "constraints.txt"),
        ]
    )

    command = module._branch_command(
        scenario=scenario,
        scenario_id="normalization",
        scenario_run_root=tmp_path / "run" / "normalization",
        matrix_dir=matrix_dir,
        args=args,
    )

    assert command[:2] == [sys.executable, str(module.BRANCH_EQUIVALENCE_SCRIPT)]
    assert command[command.index("--base-parset") + 1] == str(
        (matrix_dir / "base.parset").resolve()
    )
    assert command[command.index("--current-parset") + 1] == str(
        (matrix_dir / "current.parset").resolve()
    )
    assert command[command.index("--repeatability-repetitions") + 1] == "3"
    assert command[command.index("--repeatability-work-root") + 1] == str(
        tmp_path / "work" / module._scenario_scratch_name("normalization")
    )
    assert "--setup-base-env" in command
    assert "--base-system-site-packages" in command
    assert command[command.index("--base-pip-constraint") + 1] == str(
        (matrix_dir / "constraints.txt").resolve()
    )


def test_option_matrix_run_summarizes_reports_and_skips(tmp_path, monkeypatch):
    module = load_branch_option_matrix_script()
    matrix = tmp_path / "matrix.json"
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("base.parset", "current.parset"):
        (inputs / name).write_text("[global]\ndir_working = work\n", encoding="utf-8")
    _write_matrix(
        matrix,
        [
            {
                "id": "normalization",
                "base_parset": "inputs/base.parset",
                "current_parset": "inputs/current.parset",
                "notes": "high-impact option",
            },
            {
                "id": "screens",
                "skip_reason": "requires IDGCal in the target environment",
            },
        ],
    )

    class Completed:
        returncode = 0

    def fake_run(command, **kwargs):
        run_root = Path(command[command.index("--run-root") + 1])
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "branch-equivalence-report.json").write_text(
            json.dumps(
                {
                    "comparison": {
                        "passed": True,
                        "metrics": {"fits": 2, "h5": 1},
                        "failures": [],
                        "warnings": ["output-record optional artifact basenames differ"],
                    }
                }
            ),
            encoding="utf-8",
        )
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    run_root = tmp_path / "run"
    args = module.parse_args(["--matrix", str(matrix), "--run-root", str(run_root)])

    assert module.run(args) == 0

    summary = json.loads((run_root / "option-matrix-summary.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in summary["scenarios"]] == ["normalization", "screens"]
    assert summary["scenarios"][0]["result"] == "pass"
    assert summary["scenarios"][0]["warning_count"] == 1
    assert summary["scenarios"][1]["result"] == "skipped"
    assert summary["scenarios"][1]["skip_reason"] == "requires IDGCal in the target environment"
    markdown = (run_root / "option-matrix-summary.md").read_text(encoding="utf-8")
    assert "`normalization` | pass" in markdown
    assert "`screens` | skipped" in markdown


def test_repeatability_summary_uses_gate_decision_for_bounded_pairs():
    module = load_branch_option_matrix_script()
    report = {
        "pair_summaries": [
            {
                "pair_id": "base-rep-01_vs_base-rep-02",
                "passed": False,
                "warning_count": 0,
            },
            {
                "pair_id": "base-rep-01_vs_current-rep-01",
                "passed": False,
                "warning_count": 1,
            },
        ],
        "gate_decision": {
            "overall_status": "pass",
            "science_product_validity": {"failed_cross_pairs": []},
            "pair_statuses": {
                "base-rep-01_vs_base-rep-02": {"status": "repeatability-reference"},
                "base-rep-01_vs_current-rep-01": {"status": "repeatability-bounded"},
            },
        },
    }

    summary = module._repeatability_report_summary(report)

    assert summary["result"] == "pass"
    assert summary["pairs"] == 2
    assert summary["passed_pairs"] == 2
    assert summary["failure_count"] == 0
    assert summary["warning_count"] == 1


def test_option_matrix_can_run_one_selected_scenario(tmp_path, monkeypatch):
    module = load_branch_option_matrix_script()
    matrix = tmp_path / "matrix.json"
    for name in ("base-a.parset", "current-a.parset", "base-b.parset", "current-b.parset"):
        (tmp_path / name).write_text("[global]\ndir_working = work\n", encoding="utf-8")
    _write_matrix(
        matrix,
        [
            {
                "id": "normalization",
                "base_parset": "base-a.parset",
                "current_parset": "current-a.parset",
            },
            {
                "id": "multi-sector-mosaic",
                "base_parset": "base-b.parset",
                "current_parset": "current-b.parset",
            },
        ],
    )

    class Completed:
        returncode = 0

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        run_root = Path(command[command.index("--run-root") + 1])
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "branch-equivalence-report.json").write_text(
            json.dumps({"comparison": {"passed": True, "metrics": {}}}),
            encoding="utf-8",
        )
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    run_root = tmp_path / "run"
    args = module.parse_args(
        [
            "--matrix",
            str(matrix),
            "--scenario",
            "multi-sector-mosaic",
            "--run-root",
            str(run_root),
        ]
    )

    assert module.run(args) == 0

    summary = json.loads((run_root / "option-matrix-summary.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in summary["scenarios"]] == ["multi-sector-mosaic"]
    assert len(commands) == 1
    assert commands[0][commands[0].index("--scenario-id") + 1] == "multi-sector-mosaic"


def test_multi_sector_mosaic_option_matrix_scenario_is_defined():
    matrix_dir = OPTION_MATRIX_PATH.parent
    matrix = json.loads(OPTION_MATRIX_PATH.read_text(encoding="utf-8"))
    scenarios = {scenario["id"]: scenario for scenario in matrix["scenarios"]}

    scenario = scenarios["multi-sector-mosaic"]

    for side in ("base", "current"):
        parset = matrix_dir / scenario[f"{side}_parset"]
        strategy = parset.with_name(f"{parset.stem}_strategy.py")
        assert parset.is_file()
        assert strategy.is_file()

        parser = configparser.ConfigParser(interpolation=None)
        parser.read(parset)
        assert parser["global"]["input_ms"].endswith("prefect_demo_multisector.ms")
        assert parser["global"]["input_skymodel"].endswith("prefect_demo_multisector_true_sky.txt")
        assert parser["global"]["apparent_skymodel"].endswith(
            "prefect_demo_multisector_apparent_sky.txt"
        )
        assert parser["global"]["strategy"].endswith(
            f"inputs/{side}/multi_sector_mosaic_strategy.py"
        )
        assert parser["imaging"]["grid_nsectors_ra"] == "2"
        assert parser["imaging"]["dde_method"] == "single"
        assert parser["imaging"]["skip_corner_sectors"] == "False"


def test_initial_skymodel_regrouping_scenarios_cover_bda_interaction():
    matrix_dir = OPTION_MATRIX_PATH.parent
    matrix = json.loads(OPTION_MATRIX_PATH.read_text(encoding="utf-8"))
    scenarios = {scenario["id"]: scenario for scenario in matrix["scenarios"]}

    expected_imaging = {
        "initial-skymodel-regroup": {
            "average_visibilities": "False",
            "bda_timebase": "0.0",
            "bda_frequencybase": "0.0",
        },
        "initial-skymodel-bda-regroup": {
            "average_visibilities": "True",
            "bda_timebase": "20000.0",
            "bda_frequencybase": "20000.0",
        },
    }

    for scenario_id, imaging_settings in expected_imaging.items():
        scenario = scenarios[scenario_id]
        assert scenario["repeatability_repetitions"] == 3
        for side in ("base", "current"):
            parset = matrix_dir / scenario[f"{side}_parset"]
            parser = configparser.ConfigParser(interpolation=None)
            parser.read(parset)

            assert parser["global"].getboolean("generate_initial_skymodel")
            assert parser["global"].getboolean("regroup_input_skymodel")
            assert parser["global"]["input_skymodel"] == "None"
            assert parser["global"]["apparent_skymodel"] == "None"
            for key, value in imaging_settings.items():
                assert parser["imaging"][key] == value

            strategy = runpy.run_path(str(parset.with_name(f"{parset.stem}_strategy.py")))
            step = strategy["strategy_steps"][0]
            assert step["target_flux"] == 1.0
            assert step["max_directions"] == 1
            assert step["regroup_model"] is True
            assert step["do_calibrate"] is True

            if side == "base":
                assert step["do_fulljones_solve"] is False
                assert step["do_slowgain_solve"] is False
            else:
                assert step["calibration_strategy"] == {
                    "di": [],
                    "dd": ["fast_phase", "medium_phase"],
                }


def test_active_equivalence_scenarios_pin_matching_bda_settings():
    matrix_dir = OPTION_MATRIX_PATH.parent
    matrix = json.loads(OPTION_MATRIX_PATH.read_text(encoding="utf-8"))
    no_bda = ("False", "0.0", "0.0", "0.0", "0.0")
    production_bda = ("True", "20000.0", "20000.0", "20000.0", "20000.0")
    expected_settings = {
        "phase-only-core": no_bda,
        "dd-phase-plus-di-fulljones": no_bda,
        "normalization-rich-demo": no_bda,
        "prediction-path-image-based": no_bda,
        "prediction-path-wsclean": no_bda,
        "bda-averaging": production_bda,
        "bda-frequency-limits": production_bda,
        "initial-skymodel-regroup": (
            "False",
            "0.0",
            "0.0",
            "20000.0",
            "20000.0",
        ),
        "initial-skymodel-bda-regroup": production_bda,
    }
    assert set(expected_settings) == ACTIVE_EQUIVALENCE_SCENARIOS

    for scenario in matrix["scenarios"]:
        if scenario.get("skip_reason"):
            continue
        scenario_id = scenario["id"]
        paired_settings = []
        for side in ("base", "current"):
            parset = matrix_dir / scenario[f"{side}_parset"]
            parser = configparser.ConfigParser(interpolation=None)
            parser.read(parset)
            settings = (
                parser["imaging"]["average_visibilities"],
                parser["imaging"]["bda_timebase"],
                parser["imaging"]["bda_frequencybase"],
                parser["calibration"]["bda_timebase"],
                parser["calibration"]["bda_frequencybase"],
            )
            assert settings == expected_settings[scenario_id]
            paired_settings.append(settings)

        assert paired_settings[0] == paired_settings[1]


def test_scenario_scratch_names_are_short_stable_and_distinct():
    module = load_branch_option_matrix_script()

    phase_name = module._scenario_scratch_name("phase-only-core")
    mixed_name = module._scenario_scratch_name("dd-phase-plus-di-fulljones")

    assert phase_name == module._scenario_scratch_name("phase-only-core")
    assert phase_name != mixed_name
    assert phase_name.startswith("eq-")
    assert len(phase_name) == 11


def test_branch_command_uses_short_unique_default_repeatability_work_root(tmp_path, monkeypatch):
    module = load_branch_option_matrix_script()
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    for name in ("base.parset", "current.parset"):
        (matrix_dir / name).write_text("[global]\n", encoding="utf-8")
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda *, prefix: f"/tmp/{prefix}test")
    args = module.parse_args(
        [
            "--matrix",
            str(matrix_dir / "option-matrix.json"),
            "--run-root",
            str(tmp_path / "reports"),
            "--repeatability-repetitions",
            "3",
        ]
    )

    command = module._branch_command(
        scenario={
            "id": "frequency-bda-with-a-descriptive-report-name",
            "base_parset": "base.parset",
            "current_parset": "current.parset",
        },
        scenario_id="frequency-bda-with-a-descriptive-report-name",
        matrix_dir=matrix_dir,
        scenario_run_root=tmp_path / "reports" / "scenario",
        args=args,
    )

    assert command[command.index("--repeatability-work-root") + 1] == "/tmp/req-test"


def test_active_equivalence_scenarios_keep_rerunnable_inputs():
    matrix_dir = OPTION_MATRIX_PATH.parent
    matrix = json.loads(OPTION_MATRIX_PATH.read_text(encoding="utf-8"))
    scenarios = {scenario["id"]: scenario for scenario in matrix["scenarios"]}

    assert ACTIVE_EQUIVALENCE_SCENARIOS <= scenarios.keys()

    for scenario_id in ACTIVE_EQUIVALENCE_SCENARIOS:
        scenario = scenarios[scenario_id]
        assert "skip_reason" not in scenario
        for side in ("base", "current"):
            parset = matrix_dir / scenario[f"{side}_parset"]
            assert parset.is_file(), f"missing {scenario_id} {side} parset"

            parser = configparser.ConfigParser(interpolation=None)
            parser.read(parset)
            strategy = Path(parser["global"]["strategy"])
            strategy_resource = parset.parent / strategy.name
            assert strategy_resource.is_file(), f"missing {scenario_id} {side} strategy"

            parset_text = parset.read_text(encoding="utf-8")
            assert "/docs/source/development/science_equivalence_runs/" not in parset_text
            assert "/app/runs/" not in parset_text


def test_option_matrix_returns_failure_when_report_fails(tmp_path, monkeypatch):
    module = load_branch_option_matrix_script()
    matrix = tmp_path / "matrix.json"
    _write_matrix(
        matrix,
        [
            {
                "id": "predict",
                "base_parset": "base.parset",
                "current_parset": "current.parset",
            }
        ],
    )
    (tmp_path / "base.parset").write_text("[global]\ndir_working = work\n", encoding="utf-8")
    (tmp_path / "current.parset").write_text("[global]\ndir_working = work\n", encoding="utf-8")

    class Completed:
        returncode = 1

    def fake_run(command, **kwargs):
        run_root = Path(command[command.index("--run-root") + 1])
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "branch-equivalence-report.json").write_text(
            json.dumps(
                {
                    "comparison": {
                        "passed": False,
                        "metrics": {},
                        "failures": ["FITS image differs"],
                        "warnings": [],
                    }
                }
            ),
            encoding="utf-8",
        )
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    args = module.parse_args(["--matrix", str(matrix), "--run-root", str(tmp_path / "run")])

    assert module.run(args) == 1
