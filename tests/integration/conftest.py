"""Module for pytest fixtures."""

import configparser
import tempfile
from pathlib import Path

import pytest

REPO_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESOURCES_DIR = REPO_ROOT_DIR / "tests" / "resources"


@pytest.fixture
def generated_parset_path(request, tmp_path):
    """Fixture to generate a complete parset from a template and return the
    path.

    This fixture accepts a request.param tuple, which is used to read in and
    update a template parset file. The request param tuple should contain
    three paths to the following files:

    1. Template parset (e.g. in tests/resources/parsets/)
    2. Strategy file (e.g. in tests/integration/strategies/)
    3. True sky model (e.g. in tests/resources/)
    4. Apparent sky model (e.g. in tests/resources/)

    A new parset file is created in a temporary directory, updating the
    template parset with the provided strategy and sky model paths as well
    as setting the following:

    - `dir_working` to a temporary directory
    - `input_ms` to the test measurement set in tests/data/
    - `strategy` to the provided strategy file path
    - `apparent_skymodel` to the provided sky model path
    - `input_skymodel` to the provided sky model path
    - `photometry_skymodel` to the provided sky model path
    - `astrometry_skymodel` to the provided sky model path
    - `local_scratch_dir` to a temporary directory
    - `global_scratch_dir` to a temporary directory

    This fixture can be used to test rapthor runs end to end on a small input
    measurement set with different strategies and sky models.
    """
    parset_path, strategy_path, input_skymodel_path, apparent_skymodel_path = request.param
    parset_path = REPO_ROOT_DIR / parset_path
    strategy_path = REPO_ROOT_DIR / strategy_path
    if input_skymodel_path:
        input_skymodel_path = REPO_ROOT_DIR / input_skymodel_path
    if apparent_skymodel_path:
        apparent_skymodel_path = REPO_ROOT_DIR / apparent_skymodel_path
    input_ms_path = RESOURCES_DIR / "test.ms"

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
        strategy=str(strategy_path),
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
