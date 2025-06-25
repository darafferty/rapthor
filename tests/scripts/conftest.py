"""
This files contains the configuration for pytest, including fixtures and hooks
for this directory.
"""
import shutil
from pathlib import Path

import pytest

RESOURCE_DIR = Path(__file__).parent / ".." / "resources"

@pytest.fixture
def test_ms(tmp_path):
    """
    Fixture to provide a copy of the test MS in the resources directory.
    Yield the POSIX path to the copy of the MS.
    """
    shutil.copytree(RESOURCE_DIR / "test.ms", tmp_path / "test.ms")
    yield (tmp_path / "test.ms").as_posix()

@pytest.fixture
def soltab():
    """
    Fixture to provide a dummy soltab for testing.
    This is a placeholder and should be replaced with actual soltab creation logic.
    """
    # Create a dummy soltab or return a mock object as needed
    return "dummy_soltab"  # Replace with actual soltab creation logic if necessary
