"""Live CWL-vs-Prefect equivalence checks using existing integration fixtures."""

import configparser
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from rapthor.execution.equivalence import compare_backend_runs, format_differences

from .utils import update_parset_path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_LIVE_CWL_EQUIVALENCE_ENV = "RAPTHOR_RUN_LIVE_CWL_EQUIVALENCE"
LEGACY_CWL_REPO_ENV = "RAPTHOR_LEGACY_CWL_REPO"


def _require_legacy_cwl_repo():
    if os.environ.get(RUN_LIVE_CWL_EQUIVALENCE_ENV) != "1":
        pytest.skip(f"{RUN_LIVE_CWL_EQUIVALENCE_ENV}=1 is required")

    legacy_repo = os.environ.get(LEGACY_CWL_REPO_ENV)
    if legacy_repo in (None, ""):
        pytest.skip(f"{LEGACY_CWL_REPO_ENV} is not set")

    legacy_repo_path = Path(legacy_repo)
    if not (legacy_repo_path / "rapthor" / "process.py").is_file():
        pytest.skip(f"{LEGACY_CWL_REPO_ENV} does not point to a Rapthor checkout")
    return legacy_repo_path


def _pythonpath_with_repo(repo_root):
    current_pythonpath = os.environ.get("PYTHONPATH")
    if current_pythonpath:
        return str(repo_root) + os.pathsep + current_pythonpath
    return str(repo_root)


def _run_python(repo_root, code, parset_path, label):
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_with_repo(repo_root)
    result = subprocess.run(
        [sys.executable, "-c", code, str(parset_path)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{label} failed with output:\n{output}"
    assert "Rapthor has finished :)" in output
    return output


def _copy_parset_for_backend(source_parset, target_parset, working_dir):
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(source_parset)
    parser.set("global", "dir_working", str(working_dir))

    if parser.has_section("cluster"):
        scratch_dir = working_dir / "s"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        if parser.has_option("cluster", "local_scratch_dir"):
            parser.set("cluster", "local_scratch_dir", str(scratch_dir))
        if parser.has_option("cluster", "global_scratch_dir"):
            parser.set("cluster", "global_scratch_dir", str(scratch_dir))

    target_parset.parent.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    with target_parset.open("w") as handle:
        parser.write(handle)
    return target_parset


@pytest.mark.integration
@pytest.mark.parametrize(
    "generated_parset_path,single_loop_strategy_with_calibration_strategy",
    [
        (
            (
                "tests/resources/integration_template.parset",
                "tests/resources/integration_true_sky.txt",
                "tests/resources/integration_apparent_sky.txt",
            ),
            {"di": ["fast_phase"], "dd": []},
        )
    ],
    indirect=["generated_parset_path", "single_loop_strategy_with_calibration_strategy"],
)
def test_existing_di_fast_phase_integration_scenario_matches_cwl_and_prefect(
    generated_parset_path,
    single_loop_strategy_with_calibration_strategy,
):
    """Reuse the DI fast-phase integration scenario as a live equivalence check."""
    legacy_repo = _require_legacy_cwl_repo()
    source_parset = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_with_calibration_strategy),
        },
    )

    run_root = Path(tempfile.mkdtemp(prefix="rl-", dir="/tmp"))
    cwl_working_dir = run_root / "c" / "w"
    prefect_working_dir = run_root / "p" / "w"
    cwl_parset = _copy_parset_for_backend(
        source_parset,
        run_root / "c.parset",
        cwl_working_dir,
    )
    prefect_parset = _copy_parset_for_backend(
        source_parset,
        run_root / "p.parset",
        prefect_working_dir,
    )

    _run_python(
        legacy_repo,
        "import sys\nfrom rapthor import process\nprocess.run(sys.argv[1])\n",
        cwl_parset,
        "Legacy CWL",
    )
    _run_python(
        REPO_ROOT,
        (
            "import sys\n"
            "from rapthor.execution.flows.process import process_flow\n"
            "process_flow(sys.argv[1])\n"
        ),
        prefect_parset,
        "Prefect",
    )

    differences = compare_backend_runs(cwl_working_dir, prefect_working_dir)
    assert not differences, format_differences(differences)
    shutil.rmtree(run_root)
