"""
This files contains the configuration for pytest, including fixtures and hooks
for this directory.
"""

import configparser
import contextlib
import os
import shutil
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path

import lsmtool
import numpy as np
import pytest
import requests
from astropy.table import Table
from lsmtool.facet import read_ds9_region_file

from rapthor.lib.field import Field
from rapthor.lib.observation import Observation
from rapthor.lib.parset import parset_read
from rapthor.lib.sector import Sector

TEST_ROOT_DIR = Path(__file__).parent
REPO_ROOT_DIR = TEST_ROOT_DIR.parent
RESOURCE_DIR = TEST_ROOT_DIR / "resources"

TEST_MS_ARCHIVE_URL = "https://support.astron.nl/software/ci_data/rapthor/tDDECal.in_MS.tgz"
TEST_MS_ARCHIVE_DIRNAME = "tDDECal.MS"
TEST_MS_DIRNAME = "test.ms"
TEST_TRUE_SKYMODEL = (RESOURCE_DIR / "test_true_sky.txt").as_posix()
TEST_APPARENT_SKYMODEL = (RESOURCE_DIR / "test_apparent_sky.txt").as_posix()


# ---------------------------------------------------------------------------- #
# Config


def pytest_configure(config):
    config.resource_dir = RESOURCE_DIR


# ---------------------------------------------------------------------------- #
# Helper functions


@contextlib.contextmanager
def assert_logged(caplog, logger, level, expected_messages=(), *, expected_message=None):
    """
    Context manager for asserting the presence of specific messages in the
    captured log records.

    Since this function is decorated as a context manager, the underlying
    object created is a ContextDecorator so this function can be used as a
    as decorators as well as in with statements. It is also re-entrant, so
    the same context manager

    Parameters
    ----------
    caplog : pytest.LogCaptureFixture
        The pytest caplog fixture for capturing log records.
    logger : str
        Logger name
    level : int
        Logging level
    expected_messages : str or tuple of str, optional
        Sequence of expected messages, by default ()

    Other Parameters
    ----------------
    expected_message : str, optional
        Keyword-only parameter to support user passing a single expected
        message, by default None

    Examples
    --------
    The following example will pass, because the expected message is present in
    the logs:

    >>> with assert_logged(caplog, "my_logger", logging.INFO, expected_message="Hello World"):
    ...    logging.getLogger("my_logger").info('Hello World')

    Failure case:
    >>> with assert_logged(caplog, "my_logger", logging.INFO, expected_message="Hello World"):
    ...    logging.getLogger("my_logger").info('Nope')

    Checking for multiple messages:
    >>> with assert_logged(caplog, "my_logger", logging.INFO, expected_messages=["Hello", "World"]):
    ...    logger = logging.getLogger("my_logger")
    ...    logger.info('Hello')
    ...    logger.info('World')

    Raises
    ------
    TypeError
        If the expected_messages parameter is not a string or a sequence of strings.
    AssertionError
        If any of the expected messages are not found in the logs.
    """

    # validate inputs
    if isinstance(expected_messages, str):
        expected_messages = [expected_messages]

    if not isinstance(expected_messages, Sequence):
        raise TypeError("expected_messages parameter should be a string or a sequence of strings")

    if expected_message is not None:
        expected_messages = [*expected_messages, expected_message]

    if not expected_messages:
        raise ValueError("expected messages list is empty")

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


def _download_test_ms(destination):
    response = requests.get(TEST_MS_ARCHIVE_URL, timeout=300)
    response.raise_for_status()

    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        archive_path = tmp_dir / "test.ms.tgz"
        archive_path.write_bytes(response.content)

        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(tmp_dir)

        extracted_dir = tmp_dir / TEST_MS_ARCHIVE_DIRNAME
        if not extracted_dir.exists():
            raise FileNotFoundError(f"Downloaded archive did not contain {TEST_MS_ARCHIVE_DIRNAME}")

        try:
            shutil.move(extracted_dir.as_posix(), destination.as_posix())
        except shutil.Error:
            if not destination.exists():
                raise


def _copy_from_resource_folder_to_test_path(filename, tmp_path):
    """
    Copy test resource file to temporary test folder.

    Parameters
    ----------
    filename : str
        Name of the file in the resources folder to copy.
    tmp_path : pathlib.Path
        Path to the temporary directory where the file will be copied.

    Returns
    -------
    pathlib.Path
        Path to the copied file.
    """
    source = RESOURCE_DIR / filename
    target = tmp_path / filename
    shutil.copy(source, target)
    return target


def ensure_test_ms(resource_dir):
    destination = Path(resource_dir) / TEST_MS_DIRNAME
    if destination.exists():
        return destination

    _download_test_ms(destination)
    return destination


# ---------------------------------------------------------------------------- #
# Fixtures


@pytest.fixture(scope="session")
def test_ms(tmp_path_factory):
    """
    Fixture to provide a copy of the test MS in the resources directory.
    Yield the POSIX path to the copy of the MS.

    The test MS contains:
        - 8 channels
        - Minimum frequency = 134288024.90234375 Hz
        - Maximum frequency = 134458923.33984375 Hz
        - Channel width = 24414.0625 Hz
        - Phase center at RA = 0.4262457236387493 radians, Dec = 0.5787469737178225 radians
    """
    source = ensure_test_ms(RESOURCE_DIR)
    target = (tmp_path_factory.mktemp("test_ms") / TEST_MS_DIRNAME).as_posix()
    shutil.copytree(source, target)
    return target


@pytest.fixture
def parset(tmp_path, test_ms):
    """
    Fixture to create a parset dictionary for testing.
    """
    output_path = tmp_path / "test.parset"
    _generate_parset(
        RESOURCE_DIR / "test.parset",
        output_path=output_path,
        dir_working=tmp_path.as_posix(),
        input_ms=test_ms,
        input_skymodel=TEST_TRUE_SKYMODEL,
        apparent_skymodel=TEST_APPARENT_SKYMODEL,
    )
    return parset_read(output_path, use_log_file=False)


@pytest.fixture
def field(parset):
    """
    Fixture to create a Field object for testing.
    """
    return Field(parset)


@pytest.fixture
def sector(field):
    """Create a sector instance using the test Field and its phase center."""
    return Sector(
        name="test_sector", ra=field.ra, dec=field.dec, width_ra=1.0, width_dec=1.0, field=field
    )


@pytest.fixture
def outlier_sector(field):
    """Create an outlier sector instance using the test Field and its phase center."""
    s = Sector(
        name="outlier_sector",
        ra=field.ra,
        dec=field.dec,
        width_ra=1.0,
        width_dec=1.0,
        field=field,
    )
    s.is_outlier = True
    return s


@pytest.fixture
def sky_model_path(tmp_path):
    """
    Fixture to create an apparent SkyModel for testing.
    """
    return _copy_from_resource_folder_to_test_path("test_apparent_sky.txt", tmp_path)


@pytest.fixture
def selected_sky_model_path(tmp_path):
    """
    Fixture to create a selected apparent SkyModel for testing.
    """
    return _copy_from_resource_folder_to_test_path("test_apparent_sky_selected.txt", tmp_path)


@pytest.fixture
def single_source_sky_model(tmp_path):
    """
    Fixture to create simple sky model file with a single source.

    It returns a dictionary with the path of the file and the source values.
    """
    name = "src"
    ra = 4.0
    dec = 2.0
    reference_frequency = 142000000.0
    path = tmp_path / "single_source_sky.txt"
    path.write_text(
        "FORMAT = Name, Type, Ra, Dec, I, ReferenceFrequency\n"
        f"{name}, POINT, {ra}, {dec}, 0.042, {reference_frequency}\n"
    )
    return {
        "path": path,
        "name": name,
        "ra": ra,
        "dec": dec,
        "reference_frequency": reference_frequency,
    }


@pytest.fixture
def empty_source_sky_model(tmp_path):
    """
    Fixture to create an empty sky model file.

    It returns a dictionary with the path of the file and the source values.
    """
    path = tmp_path / "empty_sky.txt"
    path.write_text("FORMAT = Name, Type, Ra, Dec, I, ReferenceFrequency\n")
    return path


@pytest.fixture
def true_sky_path(tmp_path):
    """
    Fixture to create a true SkyModel for testing.
    """
    return _copy_from_resource_folder_to_test_path("integration_true_sky.txt", tmp_path)


@pytest.fixture
def apparent_sky_path(tmp_path):
    """
    Fixture to create an apparent SkyModel for testing.
    """
    return _copy_from_resource_folder_to_test_path("integration_apparent_sky.txt", tmp_path)


@pytest.fixture
def true_sky_model(sky_model_path):
    """
    Fixture to provide a mock sky model from a survey for testing.
    This is a placeholder and should be replaced with actual sky model creation logic.
    """
    return lsmtool.load(sky_model_path.as_posix())


@pytest.fixture
def empty_sky_model(empty_source_sky_model):
    """
    Fixture to create an empty sky model file for testing.
    """
    return lsmtool.load(empty_source_sky_model.as_posix())


@pytest.fixture
def soltab():
    """
    Fixture to provide a dummy soltab for testing.
    This is a placeholder and should be replaced with actual soltab creation logic.
    """
    # Create a dummy soltab or return a mock object as needed
    return "dummy_soltab"  # Replace with actual soltab creation logic if necessary


@pytest.fixture
def observation(test_ms):
    """
    Fixture to create an Observation object for testing.
    """
    return Observation(test_ms)


@pytest.fixture
def input_catalog_fits(tmp_path):
    """
    Fixture to provide a path for a mock input catalog FITS file.
    """
    catalog_path = tmp_path / "input_catalog.fits"
    # Create an empty file or copy a test FITS file if needed
    catalog_path.touch()
    return catalog_path


@pytest.fixture
def image_fits(tmp_path):
    """
    Fixture to provide a path for a mock image FITS file.

    Copy file from resources folder to temporary directory.
    Data adapted from fits.util.get_testdata_filepath('test0.fits'):

    .. code-block:: python

        from astropy.io import fits

        fits_image_filename = fits.util.get_testdata_filepath('test0.fits')
        with fits.open(fits_image_filename) as hdul:
            hdr = hdul[1].header
            hdr["BMAJ"] = 0.1
            hdr["BMIN"] = 0.1
            hdr["PA"] = 0.0
            hdul.writeto(tmp_path / "test_image.fits")

    """
    return _copy_from_resource_folder_to_test_path("test_image.fits", tmp_path)


@pytest.fixture
def facet_region_ds9(tmp_path):
    """
    Fixture to create a region file for testing.
    """
    return _copy_from_resource_folder_to_test_path("test.reg", tmp_path)


@pytest.fixture
def custom_strategy(tmp_path):
    """
    Fixture to create a custom strategy file for testing.
    """
    return _copy_from_resource_folder_to_test_path("custom_strategy.py", tmp_path)


def _get_test_run_root():
    """Keep CI integration runs inside the project so GitLab can upload logs."""
    if ci_project_dir := os.environ.get("CI_PROJECT_DIR"):
        # Keep the path short enough for multiprocessing AF_UNIX socket names.
        run_root = Path(ci_project_dir) / "ci" / "i"
        run_root.mkdir(parents=True, exist_ok=True)
        return run_root
    return Path("/tmp")


def _generate_parset(template_parset=None, config=None, output_path=None, **kws):
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


@pytest.fixture
def parset_for_field_test(tmp_path_factory, test_ms):
    target_path = tmp_path_factory.mktemp("test_field") / "generated.parset"
    generate_parset(
        RESOURCE_DIR / "test.parset",
        test_ms,
        target_path,
        RESOURCE_DIR / "test_true_sky.txt",
        RESOURCE_DIR / "test_apparent_sky.txt",
    )
    return parset_read(target_path)


def _make_source_catalog(n_channels=8, n_sources=8, alpha=-0.7, ref_flux=1.0, outliers=False):
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
    table = Table(columns)
    return table


@pytest.fixture
def source_catalog_fits(tmp_path):
    """
    A synthetic PyBDSF spectral-index-mode source catalog FITS file whose
    sources are centered on the test MS phase center. Contains 8 channels.
    """
    catalog_path = str(tmp_path / "test_source_catalog.fits")
    table = _make_source_catalog()
    table.write(catalog_path, format="fits", overwrite=True)
    return catalog_path


@pytest.fixture
def source_catalog_zero_channels_fits(tmp_path):
    """
    A synthetic PyBDSF spectral-index-mode source catalog FITS file whose
    sources are centered on the test MS phase center but with zero channels.
    """
    catalog_path = str(tmp_path / "test_source_catalog.fits")
    table = _make_source_catalog(n_channels=0)
    table.write(catalog_path, format="fits", overwrite=True)
    return catalog_path


@pytest.fixture
def source_catalog_zero_sources_fits(tmp_path):
    """
    A synthetic PyBDSF spectral-index-mode source catalog FITS file with zero sources.
    """
    catalog_path = str(tmp_path / "test_source_catalog.fits")
    table = _make_source_catalog(n_sources=0)
    table.write(catalog_path, format="fits", overwrite=True)
    return catalog_path


@pytest.fixture
def source_catalog_with_outliers_fits(tmp_path):
    """
    A synthetic PyBDSF spectral-index-mode source catalog FITS file whose
    sources include some that fail the radius and major axis cuts.
    """
    catalog_path = str(tmp_path / "test_source_catalog_with_outliers.fits")
    table = _make_source_catalog(
        n_sources=10, outliers=True
    )  # 10 sources, 4 of which will be outliers
    table.write(catalog_path, format="fits", overwrite=True)
    return catalog_path


@pytest.fixture()
def rendered_regions(pytestconfig):
    return pytestconfig.resource_dir / "test_image_regions_rendered.fits"


@pytest.fixture()
def facets(facet_region_ds9):
    return read_ds9_region_file(facet_region_ds9)
