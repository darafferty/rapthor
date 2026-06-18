"""
This files contains the configuration for pytest, including fixtures and hooks
for this directory.
"""

import shutil
import tarfile
import tempfile
from collections.abc import MutableMapping, Sequence
from pathlib import Path

import lsmtool
import pytest
import requests
from _pytest.fixtures import FixtureFunctionDefinition
from lsmtool.facet import read_ds9_region_file

from rapthor.lib.field import Field
from rapthor.lib.observation import Observation
from rapthor.lib.parset import parset_read
from rapthor.lib.sector import Sector
from rapthor.testing import _generate_parset, generate_parset_path, make_source_catalog

TEST_ROOT_DIR = Path(__file__).parent
REPO_ROOT_DIR = TEST_ROOT_DIR.parent
RESOURCE_DIR = TEST_ROOT_DIR / "resources"

TEST_MS_ARCHIVE_URL = "https://support.astron.nl/software/ci_data/rapthor/tDDECal.in_MS.tgz"
TEST_MS_ARCHIVE_DIRNAME = "tDDECal.MS"
TEST_MS_DIRNAME = "test.ms"


def pytest_configure(config):
    config.repo_root_dir = REPO_ROOT_DIR
    config.resource_dir = RESOURCE_DIR


@pytest.fixture
def resolve_fixture_values(request):
    """
    Helper fixture that retrieves fixture values dynamically for test case
    parameters containing direct fixture function references.

    This enables more flexible parametrization of test cases that require a mix
    of literal values and fixture values. Fixture functions are resolved to the
    required fixture values at runtime. The requested parameters may be a
    single value or a nested structure, in which case the value resolution will
    be done recursively.

    Examples
    --------
    >>> @pytest.fixture
    ... def my_fixture():
    ...     'do some setup here'
    ...     yield "fixture value"
    ...
    ... @pytest.mark.parametrize('resolve_fixture_values', [my_fixture], indirect=True)
    ... def test_something(resolve_fixture_values):
    ...     assert resolve_fixture_values == ["fixture value"]
    """
    return _get_fixture_value(request, request.param)


def _get_fixture_value(request, item):
    # If the item is a fixture function, resolve it to the fixture value
    if isinstance(item, FixtureFunctionDefinition):
        return request.getfixturevalue(item.__name__)

    if isinstance(item, MutableMapping):
        return {key: _get_fixture_value(request, subitem) for key, subitem in item.items()}

    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
        return [_get_fixture_value(request, subitem) for subitem in item]

    return item


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


@pytest.fixture
def copy_from_resource_folder_to_test_path(pytestconfig, tmp_path):
    """
    Fixture that returns a function which copies a test resource file to
    temporary test folder.
    """

    def _copy_helper(filename):
        """
        Copy helper function that copies a test resource file to temporary test
        folder.

        Parameters
        ----------
        filename : str
            Name of the file in the resources folder to copy.

        Returns
        -------
        pathlib.Path
            Path to the copied file.
        """
        source = pytestconfig.resource_dir / filename
        target = tmp_path / filename
        shutil.copy(source, target)
        return target

    return _copy_helper


def ensure_test_ms(resource_dir):
    destination = Path(resource_dir) / TEST_MS_DIRNAME
    if destination.exists():
        return destination

    _download_test_ms(destination)
    return destination


@pytest.fixture(scope="session")
def test_ms(pytestconfig, tmp_path_factory):
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
    source = ensure_test_ms(pytestconfig.resource_dir)
    target = (tmp_path_factory.mktemp("test_ms") / TEST_MS_DIRNAME).as_posix()
    shutil.copytree(source, target)
    yield target


@pytest.fixture
def parset(pytestconfig, tmp_path, test_ms):
    """
    Fixture to create a parset dictionary for testing.
    """
    output_path = tmp_path / "test.parset"
    _generate_parset(
        pytestconfig.resource_dir / "test.parset",
        output_path,
        dir_working=tmp_path.as_posix(),
        input_ms=test_ms,
        input_skymodel=(pytestconfig.resource_dir / "test_true_sky.txt").as_posix(),
        apparent_skymodel=(pytestconfig.resource_dir / "test_apparent_sky.txt").as_posix(),
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
def sky_model_path(copy_from_resource_folder_to_test_path):
    """
    Fixture to create an apparent SkyModel for testing.
    """
    return copy_from_resource_folder_to_test_path("test_apparent_sky.txt")


@pytest.fixture
def selected_sky_model_path(copy_from_resource_folder_to_test_path):
    """
    Fixture to create a selected apparent SkyModel for testing.
    """
    return copy_from_resource_folder_to_test_path("test_apparent_sky_selected.txt")


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
def true_sky_path(copy_from_resource_folder_to_test_path):
    """
    Fixture to create a true SkyModel for testing.
    """
    return copy_from_resource_folder_to_test_path("integration_true_sky.txt")


@pytest.fixture
def apparent_sky_path(copy_from_resource_folder_to_test_path):
    """
    Fixture to create an apparent SkyModel for testing.
    """
    return copy_from_resource_folder_to_test_path("integration_apparent_sky.txt")


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
def image_fits(copy_from_resource_folder_to_test_path):
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
    return copy_from_resource_folder_to_test_path("test_image.fits")


@pytest.fixture
def facet_region_ds9(copy_from_resource_folder_to_test_path):
    """
    Fixture to create a region file for testing.
    """
    return copy_from_resource_folder_to_test_path("test.reg")


@pytest.fixture
def custom_strategy(copy_from_resource_folder_to_test_path):
    """
    Fixture to create a custom strategy file for testing.
    """
    return copy_from_resource_folder_to_test_path("custom_strategy.py")


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
    parset_path = request.config.repo_root_dir / parset_path
    output_parset_path = tmp_path / "generated.parset"

    generate_parset_path(
        parset_path,
        output_parset_path,
        test_ms,
        input_skymodel_path,
        apparent_skymodel_path,
        normalization_skymodel_paths=None,
    )

    return output_parset_path


@pytest.fixture
def parset_for_field_test(pytestconfig, tmp_path_factory, test_ms):
    target = tmp_path_factory.mktemp("test_field") / "generated.parset"
    generate_parset_path(
        pytestconfig.resource_dir / "test.parset",
        target,
        test_ms,
        pytestconfig.resource_dir / "test_true_sky.txt",
        pytestconfig.resource_dir / "test_apparent_sky.txt",
    )
    return parset_read(target)


@pytest.fixture
def source_catalog_fits(tmp_path):
    """
    A synthetic PyBDSF spectral-index-mode source catalog FITS file whose
    sources are centered on the test MS phase center. Contains 8 channels.
    """
    catalog_path = str(tmp_path / "test_source_catalog.fits")
    table = make_source_catalog()
    table.write(catalog_path, format="fits", overwrite=True)
    return catalog_path


@pytest.fixture
def source_catalog_zero_channels_fits(tmp_path):
    """
    A synthetic PyBDSF spectral-index-mode source catalog FITS file whose
    sources are centered on the test MS phase center but with zero channels.
    """
    catalog_path = str(tmp_path / "test_source_catalog.fits")
    table = make_source_catalog(n_channels=0)
    table.write(catalog_path, format="fits", overwrite=True)
    return catalog_path


@pytest.fixture
def source_catalog_zero_sources_fits(tmp_path):
    """
    A synthetic PyBDSF spectral-index-mode source catalog FITS file with zero sources.
    """
    catalog_path = str(tmp_path / "test_source_catalog.fits")
    table = make_source_catalog(n_sources=0)
    table.write(catalog_path, format="fits", overwrite=True)
    return catalog_path


@pytest.fixture
def source_catalog_with_outliers_fits(tmp_path):
    """
    A synthetic PyBDSF spectral-index-mode source catalog FITS file whose
    sources include some that fail the radius and major axis cuts.
    """
    catalog_path = str(tmp_path / "test_source_catalog_with_outliers.fits")
    table = make_source_catalog(
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
