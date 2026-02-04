"""
This files contains the configuration for pytest, including fixtures and hooks
for this directory.
"""

import configparser
import shutil
from pathlib import Path

import pytest

from rapthor.lib.facet import Facet
from rapthor.lib.field import Field
from rapthor.lib.parset import parset_read

from rapthor.lib.observation import Observation

RESOURCE_DIR = Path(__file__).parent / "resources"

def pytest_configure(config):
    config.resource_dir = RESOURCE_DIR


@pytest.fixture
def test_ms(tmp_path):
    """
    Fixture to provide a copy of the test MS in the resources directory.
    Yield the POSIX path to the copy of the MS.
    """
    shutil.copytree(RESOURCE_DIR / "test.ms", tmp_path/ "test.ms")
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
    """
    image_path = tmp_path / "test_image.fits"
    # Create an empty file or copy a test FITS file if needed
    image_path.touch()
    return image_path


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
    return Facet(name="Square Facet",
                 ra=1.0,
                 dec=1.0,
                 vertices=[(0, 2.0),
                           (2.0, 2.0),
                           (2.0, 0),
                           (0, 0)])
