"""
Test cases for the `rapthor.lib.strategy` module.
"""

from pathlib import Path

import pytest
from rapthor.lib.field import Field
from rapthor.lib.parset import parset_read
from rapthor.lib.strategy import (set_image_strategy, set_selfcal_strategy,
                                  set_strategy, set_user_strategy)

RESOURCE_DIR = Path(__file__).parent / ".." / "resources"


@pytest.fixture
def parset():
    """
    Fixture to create a parset dictionary for testing.
    """
    return parset_read(RESOURCE_DIR / "test.parset")


@pytest.fixture
def field(parset):
    """
    Fixture to create a Field object for testing.
    """
    return Field(parset)


def test_set_strategy(field):
    pass


def test_set_selfcal_strategy(field):
    pass


def test_set_image_strategy(field):
    pass


def test_set_user_strategy(field):
    pass
