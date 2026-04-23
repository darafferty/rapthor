"""
This files contains the configuration for pytest, including fixtures and hooks
for this directory.
"""

import configparser
import shutil
import tarfile
import tempfile
import numpy as np
from pathlib import Path
from astropy.table import Table

import pytest
import requests

import lsmtool
from lsmtool.facet import Facet

from rapthor.lib.field import Field
from rapthor.lib.observation import Observation
from rapthor.lib.parset import parset_read

TEST_ROOT_DIR = Path(__file__).parent
REPO_ROOT_DIR = TEST_ROOT_DIR.parent
RESOURCE_DIR = TEST_ROOT_DIR / "resources"

TEST_MS_ARCHIVE_URL = "https://support.astron.nl/software/ci_data/rapthor/tDDECal.in_MS.tgz"
TEST_MS_ARCHIVE_DIRNAME = "tDDECal.MS"
TEST_MS_DIRNAME = "test.ms"
TEST_TRUE_SKYMODEL = (RESOURCE_DIR / "test_true_sky.txt").as_posix()
TEST_APPARENT_SKYMODEL = (RESOURCE_DIR / "test_apparent_sky.txt").as_posix()


def pytest_configure(config):
    config.resource_dir = RESOURCE_DIR


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
    yield target


@pytest.fixture
def parset(test_ms, tmp_path):
    """
    Fixture to create a parset dictionary for testing.
    """
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(RESOURCE_DIR / "test.parset")
    cfg.set("global", "dir_working", tmp_path.as_posix())
    cfg.set("global", "input_ms", test_ms)
    cfg.set("global", "input_skymodel", TEST_TRUE_SKYMODEL)
    cfg.set("global", "apparent_skymodel", TEST_APPARENT_SKYMODEL)

    parset_file = tmp_path / "test.parset"
    with parset_file.open("w") as fh:
        cfg.write(fh)

    yield parset_read(parset_file, use_log_file=False)


@pytest.fixture
def field(parset):
    """
    Fixture to create a Field object for testing.
    """
    return Field(parset)


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
def true_sky_path(tmp_path):
    """
    Fixture to create a true SkyModel for testing.
    """
    shutil.copy((RESOURCE_DIR / "integration_true_sky.txt"), tmp_path / "integration_true_sky.txt")
    return Path(tmp_path / "integration_true_sky.txt")


@pytest.fixture
def true_sky_model(sky_model_path):
    """
    Fixture to provide a mock sky model from a survey for testing.
    This is a placeholder and should be replaced with actual sky model creation logic.
    """
    return lsmtool.load(sky_model_path.as_posix())


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


def generate_parset(
    template_parset_path, input_ms, input_skymodel_path=None, apparent_skymodel_path=None
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
        input_ms=str(input_ms),
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
    return parset


def generate_parset_path(
    template_path, output_path, test_ms, input_skymodel_path, apparent_skymodel_path
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
    parset = generate_parset(parset_path, test_ms, input_skymodel_path, apparent_skymodel_path)

    with output_path.open("w") as fp:
        parset.write(fp)


@pytest.fixture
def generated_parset_path(request, tmp_path, test_ms):
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
    parset_path, input_skymodel_path, apparent_skymodel_path = request.param
    parset_path = REPO_ROOT_DIR / parset_path
    output_parset_path = tmp_path / "generated.parset"

    generate_parset_path(
        parset_path,
        output_parset_path,
        test_ms,
        input_skymodel_path,
        apparent_skymodel_path,
    )

    return output_parset_path


@pytest.fixture
def parset_for_field_test(tmp_path_factory, test_ms):
    target = tmp_path_factory.mktemp("test_field") / "generated.parset"
    generate_parset_path(
        RESOURCE_DIR / "test.parset",
        target,
        test_ms,
        RESOURCE_DIR / "test_true_sky.txt",
        RESOURCE_DIR / "test_apparent_sky.txt",
    )
    return parset_read(target)


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
        columns["RA"][3] = columns["RA"][4] + 0.005 # Sources 3 and 4: inside neighbor_cut distance
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
def source_catalog_with_outliers_fits(tmp_path):
    """
    A synthetic PyBDSF spectral-index-mode source catalog FITS file whose
    sources include some that fail the radius and major axis cuts.
    """
    catalog_path = str(tmp_path / "test_source_catalog_with_outliers.fits")
    table = _make_source_catalog(n_sources=10, outliers=True)  # 10 sources, 4 of which will be outliers
    table.write(catalog_path, format="fits", overwrite=True)
    return catalog_path