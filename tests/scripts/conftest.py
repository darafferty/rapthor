"""
This files contains the configuration for pytest, including fixtures and hooks
for this directory.
"""

import shutil
from pathlib import Path

import pytest

RESOURCE_DIR = Path(__file__).parent.parent / "resources"


@pytest.fixture
def soltab():
    """
    Fixture to provide a dummy soltab for testing.
    This is a placeholder and should be replaced with actual soltab creation logic.
    """
    # Create a dummy soltab or return a mock object as needed
    return "dummy_soltab"  # Replace with actual soltab creation logic if necessary


@pytest.fixture
def sky_model_path(tmp_path):
    """
    Fixture to create an apparent SkyModel for testing.
    """
    shutil.copy((RESOURCE_DIR / "test_apparent_sky.txt"), tmp_path / "test_apparent_sky.txt")
    return Path(tmp_path / "test_apparent_sky.txt")