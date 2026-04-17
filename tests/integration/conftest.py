"""Module for pytest fixtures."""

import configparser
import tempfile
from pathlib import Path

import pytest
from tests.conftest import ensure_test_ms

REPO_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESOURCES_DIR = REPO_ROOT_DIR / "tests" / "resources"


COMMON_STRATEGY_SETTINGS = {
    "channel_width_hz": 195312.5,
    # Set slow-gain and fulljones solves to False except when required
    "do_slowgain_solve": False,
    "do_fulljones_solve": False,
    # Don't remove bright outliers or in-field sources -- image full field
    "peel_outliers": False,
    "peel_bright_sources": False,
    # Fast phase (ionosphere) and slow gain (beam) time intervals (s)
    "fast_timestep_sec": 32.0,
    "medium_timestep_sec": 120.0,
    "slow_timestep_sec": 600.0,
    # Turn off flux-scale bootstrapping
    "do_normalize": False,
    # PyBDSF settings
    "auto_mask": 5.0,
    "auto_mask_nmiter": 2,
    "threshisl": 3.0,
    "threshpix": 5.0,
    # Constrain max nr of imaging major cycles
    "max_nmiter": 12,
    # Disable regrouping of sky model
    "regroup_model": True,
    # Max distance allowed between selected DDE calibrators
    "max_distance": None,  # no distance constraint
    # Don't check for self-cal convergence
    "do_check": False,
    "target_flux": 0.3,
    "max_directions": 4,
}


def make_step(**overrides):
    """Helper to create a strategy step with settings and overrides."""
    return {**COMMON_STRATEGY_SETTINGS, **overrides}


@pytest.fixture
def generated_parset_path(request, tmp_path):
    """Fixture to generate a complete parset from a template and return the
    path.

    This fixture is used to read in and update a template parset file. It is
    parametrised using the pytest request fixture and expects a tuple
    containing three paths to the following files:

    1. Template parset (e.g. in tests/resources/parsets/)
    2. True sky model (e.g. in tests/resources/)
    3. Apparent sky model (e.g. in tests/resources/)

    A new parset file is created in a temporary directory, updating the
    template parset with the provided strategy and sky model paths as well
    as setting the following:

    - `dir_working` to a temporary directory
    - `input_ms` to the test measurement set in tests/data/
    - `apparent_skymodel` to the provided sky model path
    - `input_skymodel` to the provided sky model path
    - `photometry_skymodel` to the provided sky model path
    - `astrometry_skymodel` to the provided sky model path
    - `local_scratch_dir` to a temporary directory
    - `global_scratch_dir` to a temporary directory

    This fixture can be used to test rapthor runs end to end on a small input
    measurement set with different strategies and sky models.
    """
    parset_path, input_skymodel_path, apparent_skymodel_path = request.param
    parset_path = REPO_ROOT_DIR / parset_path
    if input_skymodel_path:
        input_skymodel_path = REPO_ROOT_DIR / input_skymodel_path
    if apparent_skymodel_path:
        apparent_skymodel_path = REPO_ROOT_DIR / apparent_skymodel_path
    input_ms_path = ensure_test_ms(RESOURCES_DIR)

    # Keep runtime paths short to avoid AF_UNIX socket path length limits
    # in multiprocessing-based tooling (e.g. PyBDSF).
    run_dir = Path(tempfile.mkdtemp(prefix="ical-", dir="/tmp"))
    work_dir = run_dir / "work"
    scratch_dir = run_dir / "scratch"
    work_dir.mkdir()
    scratch_dir.mkdir()

    parset = configparser.ConfigParser()
    parset.read(parset_path)
    parset["global"].update(
        dir_working=str(work_dir),
        input_ms=str(input_ms_path),
    )
    if input_skymodel_path:
        parset["global"]["input_skymodel"] = str(input_skymodel_path)
        parset["imaging"]["photometry_skymodel"] = str(input_skymodel_path)
        parset["imaging"]["astrometry_skymodel"] = str(input_skymodel_path)
    if apparent_skymodel_path:
        parset["global"]["apparent_skymodel"] = str(apparent_skymodel_path)
    parset["cluster"].update(
        local_scratch_dir=str(scratch_dir),
        global_scratch_dir=str(scratch_dir),
    )

    path_to_generated_parset = tmp_path / "generated.parset"
    with path_to_generated_parset.open("w") as fp:
        parset.write(fp)
    return path_to_generated_parset


@pytest.fixture
def single_loop_strategy_path(tmp_path):
    """Fixture to generate a strategy file for a single self-calibration loop."""
    strategy_steps = [make_step(do_calibrate=True, do_image=True)]
    strategy_content = f"strategy_steps = {strategy_steps}"
    strategy_path = tmp_path / "single_loop_strategy.py"
    strategy_path.write_text(strategy_content)
    return strategy_path
