"""
This module contains helper code for testing rapthor.
"""

import configparser
import os
import tempfile
from pathlib import Path

import numpy as np
from astropy.table import Table

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


def make_source_catalog(n_channels=8, n_sources=8, alpha=-0.7, ref_flux=1.0, outliers=False):
    """
    Build a minimal synthetic PyBDSF spectral-index-mode source catalog.

    Sources are placed on a small grid around the MS phase center so that
    they pass the radius, major-axis, and neighbor-distance cuts used by
    ``main()``.
    """
    # Frequencies of the test MS (tests/resources/test.ms), 8 channels ~134 MHz
    ms_channel_frequencies = (
        np.arange(1.34288025e08, 1.34458923e08, (1.34458923e08 - 1.34288025e08) / n_channels)
        if n_channels > 0
        else np.array([])
    )

    # Phase center of the test MS in degrees (RA, Dec)
    ra0, dec0 = (24.422081, 33.159759)

    # Number of channels
    n_chan = len(ms_channel_frequencies)

    ref_freq = (
        ms_channel_frequencies[n_chan // 2] if n_chan > 0 else 1.0
    )  # Use middle channel as reference frequency, or 1.0 if no channels

    # Place sources on a regular grid with ~0.3 deg spacing (well within
    # radius_cut=3 deg and well above neighbor_cut=30/3600 deg)
    step = 0.3  # degrees
    offsets = np.arange(-(n_sources // 2), n_sources - (n_sources // 2)) * step
    source_ra = ra0 + offsets
    source_dec = np.full(n_sources, dec0)

    # Add source outside the radius cut for testing
    source_ra[0] = ra0 + 4.0  # 4 degrees, which is outside the radius_cut of 3 degrees

    # Assign power-law SEDs with slight per-source flux variation
    base_fluxes = ref_flux * (1.0 + 0.1 * np.arange(n_sources))

    # Build the column data
    columns = {
        "RA": source_ra.astype(np.float32),
        "DEC": source_dec.astype(np.float32),
        "Total_flux": base_fluxes.astype(np.float32),
        "E_Total_flux": (base_fluxes * 0.05).astype(np.float32),
        # Small deconvolved major axis — well below major_axis_cut=30/3600 deg
        "DC_Maj": np.full(n_sources, 5.0 / 3600.0, dtype=np.float32),
    }

    # Per-channel fluxes and errors
    for ch, freq in enumerate(ms_channel_frequencies, start=1):
        ch_flux = base_fluxes * (freq / ref_freq) ** alpha
        columns[f"Total_flux_ch{ch}"] = ch_flux.astype(np.float32)
        columns[f"E_Total_flux_ch{ch}"] = (ch_flux * 0.05).astype(np.float32)
        columns[f"Freq_ch{ch}"] = np.full(n_sources, freq, dtype=np.float64)

    # Add some outliers that fail the major axis and radius cuts for testing
    if n_sources >= 10 and outliers:
        columns["DC_Maj"][2] = 0.02  # Source 2: above the major_axis_cut of 0.01 degrees
        columns["RA"][3] = columns["RA"][4] + 0.005  # Sources 3 and 4: inside neighbor_cut distance
        columns["DEC"][3] = columns["DEC"][4]

    return Table(columns)
