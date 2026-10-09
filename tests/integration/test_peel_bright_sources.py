"""Integration tests for the Rapthor pipeline when using the peel_bright_sources option."""

import subprocess

import pytest

from .utils import find_command_records, get_working_dir_from_parset, update_parset_path


@pytest.mark.integration
@pytest.mark.parametrize(
    "generated_parset_path",
    [
        (
            "tests/resources/integration_template.parset",
            "tests/resources/integration_true_sky.txt",
            "tests/resources/integration_apparent_sky.txt",
        )
    ],
    indirect=True,
)
def test_rapthor_run_single_loop_peel_bright_sources(
    generated_parset_path, single_loop_strategy_path_peel_bright_sources
):
    """Test a single self-calibration loop with peel_bright_sources enabled and disabled."""
    strategy_path, peel_bright_sources = single_loop_strategy_path_peel_bright_sources

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(strategy_path),
        },
    )

    command = ["rapthor", str(updated_parset_path)]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"Rapthor failed with output:\n{output}"
    assert "Operation calibrate_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output

    working_dir = get_working_dir_from_parset(updated_parset_path)
    restore_commands = find_command_records(
        working_dir,
        operation="image_1",
        executable="wsclean",
        contains="-restore-list",
    )

    if peel_bright_sources:
        assert len(restore_commands) >= 2, (
            "Expected PB and non-PB WSClean restore commands when peeling is enabled"
        )
    else:
        assert restore_commands == [], (
            "WSClean restore commands should not run when bright-source peeling is disabled"
        )

    assert "Rapthor has finished :)" in output
