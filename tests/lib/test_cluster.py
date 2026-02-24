"""
Test cases for the cluster module in `rapthor/lib`.
"""

import numpy
import pytest
from rapthor.lib.cluster import get_available_memory


def test_get_available_memory(monkeypatch):
    """
    Test the get_available_memory function to ensure it returns the expected value.
    """
    # Mock the subprocess.getoutput to return a fixed string
    monkeypatch.setattr(
        "subprocess.getoutput", lambda _:
        "               total        used        free      shared  buff/cache   available\n"
        "Mem:              15           9           3           2           5           6\n"
        "Swap:             19           3          16\n"
        "Total:            35          12          20\n",
    )
    available_memory = get_available_memory()
    assert available_memory == 6, (
        f"Expected 6 GB of available memory, got {available_memory} GB"
    )
