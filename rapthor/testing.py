"""
This module contains helper code for testing rapthor.
"""

import configparser
import os
import tempfile
from pathlib import Path

REPO_ROOT_DIR = Path(__file__).parent.parent


def _get_test_run_root():
    """Keep CI integration runs inside the project so GitLab can upload logs."""
    if ci_project_dir := os.environ.get("CI_PROJECT_DIR"):
        # Keep the path short enough for multiprocessing AF_UNIX socket names.
        run_root = Path(ci_project_dir) / "ci" / "i"
        run_root.mkdir(parents=True, exist_ok=True)
        return run_root
    return Path("/tmp")


def generate_parset(
    template_parset_path,
    input_ms,
    input_skymodel_path=None,
    apparent_skymodel_path=None,
    normalization_skymodel_paths=None,
):
    """
    Generate a complete parset from a template, optionally update the input
    skymodel paths and return the parset as a configparser.ConfigParser object.

    This function creates a temporary working folder and scratch folder, and
    updates the provided template parset with
     - `dir_working` to a temporary directory
     - `local_scratch_dir` to a temporary directory
     - `global_scratch_dir` to a temporary directory

    If either skymodel is provided, the following keys in the parset will be
    updated:
     - `input_skymodel` in [global] to the provided sky model path
     - `apparent_skymodel` in [global] to the provided sky model path
     - `photometry_skymodel` in [imaging] to the provided sky model path
     - `astrometry_skymodel` in [imaging] to the provided sky model path

    Parameters
    ----------
    template_parset_path : str
        Path to the template parset file.
    input_ms : str
        Path to the input measurement set to set in the parset.
    input_skymodel_path : str, optional (default=None)
        Path to the input skymodel file to set in the parset.
    apparent_skymodel_path : str, optional (default=None)
        Path to the apparent skymodel file to set in the parset.
    normalization_skymodel_paths : list of str, optional (default=None)
        List of paths to the normalization skymodel files to set in the parset.

    Returns
    -------
    configparser.ConfigParser
        The updated parset as a ConfigParser object.
    """
    parset_path = REPO_ROOT_DIR / template_parset_path
    if input_skymodel_path:
        input_skymodel_path = REPO_ROOT_DIR / input_skymodel_path
    if apparent_skymodel_path:
        apparent_skymodel_path = REPO_ROOT_DIR / apparent_skymodel_path
    if normalization_skymodel_paths:
        normalization_skymodel_paths = [
            REPO_ROOT_DIR / path for path in normalization_skymodel_paths
        ]

    # Keep runtime paths short to avoid AF_UNIX socket path length limits
    # in multiprocessing-based tooling (e.g. PyBDSF). In CI, place runs under
    # the project directory so the generated logs can be collected as artifacts.
    run_dir = Path(tempfile.mkdtemp(prefix="ical-", dir=_get_test_run_root()))
    work_dir = run_dir / "work"
    scratch_dir = run_dir / "scratch"
    work_dir.mkdir()
    scratch_dir.mkdir()

    parset = configparser.ConfigParser()
    parset.read(parset_path)
    parset["global"].update(
        dir_working=str(work_dir),
        input_ms=str(input_ms),
    )
    if input_skymodel_path:
        parset["global"]["input_skymodel"] = str(input_skymodel_path)
        parset["imaging"]["photometry_skymodel"] = str(input_skymodel_path)
        parset["imaging"]["astrometry_skymodel"] = str(input_skymodel_path)
    if apparent_skymodel_path:
        parset["global"]["apparent_skymodel"] = str(apparent_skymodel_path)
    if normalization_skymodel_paths:
        parset["imaging"]["normalization_skymodels"] = (
            "["
            + ", ".join([str(path) for path in normalization_skymodel_paths if path is not None])
            + "]"
        )
        parset["imaging"]["normalization_reference_frequencies"] = (
            "["
            + ", ".join(
                [
                    str(120000000.0 + i * 60000000.0)
                    for i, _ in enumerate(normalization_skymodel_paths)
                    if _ is not None
                ]
            )
            + "]"
        )
    else:
        parset["imaging"]["normalization_reference_frequencies"] = "None"
    parset["cluster"].update(
        local_scratch_dir=str(scratch_dir),
        global_scratch_dir=str(scratch_dir),
    )
    return parset


def generate_parset_path(
    template_path,
    output_path,
    test_ms,
    input_skymodel_path,
    apparent_skymodel_path,
    normalization_skymodel_paths=None,
):
    """
    Fixture to generate a complete parset from a template and return the path.

    This fixture is used to read in and update a template parset file. It is
    parametrised using the pytest request fixture and expects a tuple
    containing three paths to the following files:

    1. Template parset (e.g. in tests/resources/parsets/)
    2. True sky model (e.g. in tests/resources/)
    3. Apparent sky model (e.g. in tests/resources/)

    This fixture can be used to test rapthor runs end to end on a small input
    measurement set with different strategies and sky models.
    For further details see `generate_parset` function.
    """
    parset_path = REPO_ROOT_DIR / template_path
    parset = generate_parset(
        parset_path,
        test_ms,
        input_skymodel_path,
        apparent_skymodel_path,
        normalization_skymodel_paths,
    )

    with output_path.open("w") as fp:
        parset.write(fp)
