"""Integration tests for the Rapthor pipeline using different calibration options."""

import subprocess

import pytest

from .utils import get_working_dir_from_parset, update_parset_path


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
def test_rapthor_run_single_loop_calibrate_di(
    generated_parset_path, single_loop_strategy_path_calibrate_di
):
    """Test a single self-calibration loop with DI calibration end to end."""

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "reweight": "False",
            "strategy": str(single_loop_strategy_path_calibrate_di),
        },
    )

    working_dir = get_working_dir_from_parset(updated_parset_path)
    print("---Rapthor working dir: ", working_dir)

    command = ["rapthor", str(updated_parset_path)]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"Rapthor failed with output:\n{output}"
    assert "Operation calibrate_di_1 completed" in output
    assert "Operation predict_di_1 completed" in output
    assert "Operation predict_1 completed" not in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    assert "Rapthor has finished :)" in output
