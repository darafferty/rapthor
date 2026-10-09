"""Tests for rapthor CLI."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("help_option", ["--help", "-h", ""])
@pytest.mark.integration
def test_rapthor_help(help_option):
    """Test the Rapthor pipeline CLI options."""
    command = [sys.executable, "-m", "rapthor.cli"]
    if help_option:
        command.append(help_option)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage: rapthor <parset>" in result.stdout
    assert "Options:" in result.stdout
