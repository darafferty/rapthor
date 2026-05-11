"""Integration tests for the shared_facet_rw feature."""

import subprocess

import pytest

from .utils import update_parset_path, find_step_workdir, get_working_dir_from_parset


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
def test_rapthor_generates_diagnostics_per_facets(generated_parset_path, single_loop_strategy_path):
    """Test a single self-calibration loop end to end."""

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path),
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
    assert "Rapthor has finished :)" in output
    print(get_working_dir_from_parset(update_parset_path))
    work_dir = find_step_workdir(update_parset_path, "image_1")
    
    
    diagnostics_file = work_dir / "sector_diagnostics.json"
    assert diagnostics_file.exists()
    assert False