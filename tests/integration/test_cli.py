"""Tests for rapthor CLI."""

import pytest
import shlex
import subprocess


@pytest.mark.parametrize("help_option", ["--help", "-h", ""])
@pytest.mark.integration
def test_rapthor_help(help_option):
    """Test the Rapthor pipeline CLI options."""
    command = shlex.split(f"rapthor {help_option}")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage: rapthor <parset>" in result.stdout
    assert "Options:" in result.stdout
