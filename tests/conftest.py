"""
This files contains the configuration for pytest, including fixtures and hooks
for this directory.
"""

import configparser
import shutil
import tarfile
import tempfile
from pathlib import Path

import pytest
import requests

from rapthor.lib.facet import Facet
from rapthor.lib.field import Field
from rapthor.lib.parset import parset_read


TEST_ROOT_DIR = Path(__file__).parent
REPO_ROOT_DIR = TEST_ROOT_DIR.parent
RESOURCE_DIR = TEST_ROOT_DIR / "resources"

TEST_MS_ARCHIVE_URL = "https://support.astron.nl/software/ci_data/rapthor/tDDECal.in_MS.tgz"
TEST_MS_ARCHIVE_DIRNAME = "tDDECal.MS"
TEST_MS_DIRNAME = "test.ms"


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


def ensure_test_ms(resource_dir):
    resource_dir = Path(resource_dir)
    destination = resource_dir / TEST_MS_DIRNAME
    if destination.exists():
        return destination

    _download_test_ms(destination)
    return destination


def pytest_configure(config):
    config.resource_dir = RESOURCE_DIR


@pytest.fixture
def test_ms(tmp_path):
    """
    Fixture to provide a copy of the test MS in the resources directory.
    Yield the POSIX path to the copy of the MS.
    """
    shutil.copytree(ensure_test_ms(RESOURCE_DIR), tmp_path / "test.ms")
    yield (tmp_path / "test.ms").as_posix()


@pytest.fixture
def parset(test_ms, tmp_path):
    """
    Fixture to create a parset dictionary for testing.
    """
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(RESOURCE_DIR / "test.parset")
    cfg.set("global", "dir_working", tmp_path.as_posix())
    cfg.set("global", "input_ms", test_ms)
    cfg.set("global", "input_skymodel", (RESOURCE_DIR / "test_true_sky.txt").as_posix())
    cfg.set("global", "apparent_skymodel", (RESOURCE_DIR / "test_apparent_sky.txt").as_posix())

    parset_file = tmp_path / "test.parset"
    with parset_file.open("w") as fh:
        cfg.write(fh)

    parset_dict = parset_read(parset_file, use_log_file=False)
    yield parset_dict


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
    shutil.copy((RESOURCE_DIR / "test_apparent_sky.txt"), tmp_path / "test_apparent_sky.txt")
    return Path(tmp_path / "test_apparent_sky.txt")


@pytest.fixture
def selected_sky_model_path(tmp_path):
    """
    Fixture to create a selected apparent SkyModel for testing.
    """
    shutil.copy(
        (RESOURCE_DIR / "test_apparent_sky_selected.txt"),
        tmp_path / "test_apparent_sky_selected.txt",
    )
    return Path(tmp_path / "test_apparent_sky_selected.txt")


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
    shutil.copy((RESOURCE_DIR / "test_image.fits"), tmp_path / "test_image.fits")
    return Path(tmp_path / "test_image.fits")


@pytest.fixture
def facet_region_ds9(tmp_path):
    """
    Fixture to create a region file for testing.
    """
    shutil.copy((RESOURCE_DIR / "test.reg"), tmp_path / "test.reg")
    return Path(tmp_path / "test.reg")


@pytest.fixture
def facet():
    """
    Fixture to create a facet for testing.
    """
    return Facet(
        name="Square Facet", ra=1.0, dec=1.0, vertices=[(0, 2.0), (2.0, 2.0), (2.0, 0), (0, 0)]
    )


@pytest.fixture
def custom_strategy(tmp_path):
    """
    Fixture to create a region file for testing.
    """


def generate_parset(template_parset_path, input_ms, input_skymodel_path=None, apparent_skymodel_path=None):
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


def generate_parset_path(template_path, output_path, test_ms, input_skymodel_path, apparent_skymodel_path):
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
        apparent_skymodel_path
    )

    return output_parset_path

