"""
This files contains the configuration for pytest, including fixtures and hooks
for this directory.
"""

import shutil
from pathlib import Path

import pytest

RESOURCE_DIR = Path(__file__).parent / "resources"

def pytest_configure(config):
    config.resource_dir = RESOURCE_DIR


@pytest.fixture
def test_ms(tmp_path):
    """
    Fixture to provide a copy of the test MS in the resources directory.
    Yield the POSIX path to the copy of the MS.
    """
    shutil.copytree(RESOURCE_DIR / "test.ms", tmp_path / "test.ms")
    yield (tmp_path / "test.ms").as_posix()
