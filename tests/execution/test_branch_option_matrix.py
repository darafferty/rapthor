import configparser
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "dev" / "run_branch_option_matrix.py"
OPTION_MATRIX_PATH = (
    Path(__file__).parents[2]
    / "docs"
    / "source"
    / "development"
    / "science_equivalence_runs"
    / "2026-07-06-option-matrix"
    / "option-matrix.json"
)


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
        tmp_path / "work" / "normalization"
    )
    assert "--setup-base-env" in command
    assert "--base-system-site-packages" in command


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
