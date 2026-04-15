"""Integration tests for the Rapthor pipeline."""

import pytest
import subprocess
import sys
from pathlib import Path

@pytest.mark.integration
def test_rapthor_pipeline():
    """Test the Rapthor pipeline end-to-end."""
    assert True  # Placeholder for checking CI/CD setup

@pytest.mark.integration
def test_rapthor_help():
    """Test the Rapthor pipeline CLI options."""
    command = ["rapthor", "--help"]
    
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage: rapthor <parset>" in result.stdout
    assert "Options:" in result.stdout