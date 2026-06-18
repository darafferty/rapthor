"""
This module contains helper code for testing rapthor.
"""

import configparser
import contextlib
import os
import tempfile
from pathlib import Path

import numpy as np
from astropy.table import Table

REPO_ROOT_DIR = Path(__file__).parent.parent


@contextlib.contextmanager
def assert_logged(caplog, logger, level, *expected_messages):
    """
    Context manager for asserting the presence of specific messages in the
    captured log records.

    Since this function is decorated as a context manager, the underlying
    object created is a ContextDecorator so this function can be used as a
    decorator as well as in with statements. It is also re-entrant, so the
    same context manager can be reused multiple times.

    Parameters
    ----------
    caplog : pytest.LogCaptureFixture
        The pytest caplog fixture for capturing log records.
    logger : str
        Logger name
    level : int
        Logging level
    *expected_messages
        One or more expected messages to scan the logs for.


    Examples
    --------
    The following example will pass, because the expected message is present in
    the logs:

    >>> with assert_logged(caplog, "my_logger", logging.INFO, "Hello World"):
    ...    logging.getLogger("my_logger").info('Hello World')

    Failure case:
    >>> with assert_logged(caplog, "my_logger", logging.INFO, "Hello World"):
    ...    logging.getLogger("my_logger").info('Nope')

    Checking for multiple messages:
    >>> with assert_logged(caplog, "my_logger", logging.INFO, "Hello", "World"):
    ...    logger = logging.getLogger("my_logger")
    ...    logger.info('Hello')
    ...    logger.info('World')

    Raises
    ------
    TypeError
        If any of the expected_messages are not strings.
    AssertionError
        If any of the expected messages are not found in the logs.
    """

    # validate inputs
    if not expected_messages:
        raise ValueError("At least one expected message must be provided")

    for msg in expected_messages:
        if not isinstance(msg, str):
            raise TypeError(f"expected message must be a str, not {type(msg).__name__}")

    # Capture logs at the level for the given logger
    with caplog.at_level(level, logger=logger) as context:
        yield context

        for expected_message in expected_messages:
            if expected_message not in caplog.text:
                raise AssertionError(
                    f"Expected message(s) not found in logs:\n\t{expected_message!r}"
                )


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
    output_path=None,
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
    These temporary direcories are created if they do not exist. 

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
            str(REPO_ROOT_DIR / path) for path in normalization_skymodel_paths if path
        ]

    # Keep runtime paths short to avoid AF_UNIX socket path length limits
    # in multiprocessing-based tooling (e.g. PyBDSF). In CI, place runs under
    # the project directory so the generated logs can be collected as artifacts.
    run_dir = Path(tempfile.mkdtemp(prefix="ical-", dir=_get_test_run_root()))
    work_dir = run_dir / "work"
    scratch_dir = run_dir / "scratch"
    work_dir.mkdir()
    scratch_dir.mkdir()

    config = {
        "global": {
            "dir_working": work_dir,
            "input_ms": input_ms,
        },
        "cluster": {
            "local_scratch_dir": scratch_dir,
            "global_scratch_dir": scratch_dir,
        },
        "imaging": {},
    }
    if input_skymodel_path:
        config["global"]["input_skymodel"] = input_skymodel_path
        config["imaging"]["photometry_skymodel"] = input_skymodel_path
        config["imaging"]["astrometry_skymodel"] = input_skymodel_path

    if apparent_skymodel_path:
        config["global"]["apparent_skymodel"] = apparent_skymodel_path

    if normalization_skymodel_paths:
        config["imaging"]["normalization_skymodels"] = (
            f"[{', '.join(normalization_skymodel_paths)}]"
        )
        ref_freq = 1.42e8 + np.arange(len(normalization_skymodel_paths)) * 1e3
        config["imaging"]["normalization_reference_frequencies"] = (
            f"[{', '.join(ref_freq.astype(str))}]"
        )
    else:
        config["imaging"]["normalization_reference_frequencies"] = "None"

    return _generate_parset(parset_path, config, output_path)


def _generate_parset(template_parset=None, config=None, output_path=None, **kws):
    """
    Base function to generate a parset from a template and a config dictionary,
    optionally writing the result to an output path.

    Parameters
    ----------
    template_parset : configparser.ConfigParser or str or Path, optional
        Template parset to use as a base, by default None. If no template is
        provided, the parset will be initialized as an empty ConfigParser
        object.
    config : dict, optional
        Configuration dictionary to update the parset, by default None. If no
        config is provided, the input template_parset must be provided.
    output_path : str or Path, optional
        Path to write the generated parset, by default None. If not provided,
        the parset will not be written to disk.
    kws : dict, optional
        Additional keyword arguments are added to the global section of the
        parset.

    Returns
    -------
    configparser.ConfigParser
        The generated parset as a ConfigParser object.

    Raises
    ------
    TypeError
        If invalid input types are provided for template_parset, config, or
        output_path.
    """

    if isinstance(config, Path):
        raise TypeError()

    parset = configparser.ConfigParser()
    if isinstance(template_parset, configparser.ConfigParser):
        parset = template_parset
    elif isinstance(template_parset, (str, Path)):
        parset.read(template_parset)
    elif template_parset is not None:
        raise TypeError(
            "Invalid type for template_parset. Expected str, Path, or ConfigParser.",
        )

    config = config or {}
    if kws:
        config["global"] = config.get("global", {}) | kws

    for section, options in config.items():
        if section is not None and section not in parset:
            parset.add_section(section)

        for option, value in options.items():
            parset.set(section, str(option), str(value))

    if output_path:
        with Path(output_path).open("w") as fp:
            parset.write(fp)

    return parset


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
        # Sources 3 and 4: inside neighbor_cut distance
        columns["RA"][3] = columns["RA"][4] + 0.005
        columns["DEC"][3] = columns["DEC"][4]

    return Table(columns)
