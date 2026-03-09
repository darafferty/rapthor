"""Test CWL command line generation."""
from pathlib import Path

import pytest

from ..cwl_cmdline import generate_command_line


@pytest.fixture
def simple_tool_cwl():
    """Path to simple_tool.cwl test fixture."""
    return Path(__file__).parent / "test_data" / "real_cwl_test" / "simple_tool.cwl"


def test_simple_tool_command_line(simple_tool_cwl):
    """Test that simple_tool.cwl generates the expected command line."""
    cmd = generate_command_line(simple_tool_cwl)
    
    assert cmd[0] == "bash"
    assert cmd[1] == "-c"
    assert "Hello CWL" in cmd[2]
    assert "echo" in cmd[2]
