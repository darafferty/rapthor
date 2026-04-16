"""Integration tests for the Rapthor pipeline."""

import configparser
import pytest
import subprocess


def update_parset_path(parset_path, param_dict):
    """Helper function to update parset parameters and return a new path."""
    parset = configparser.ConfigParser()
    parset.read(parset_path)

    for section in parset.sections():
        for key, value in param_dict.items():
            if key in parset[section]:
                parset[section][key] = value

    updated_parset_path = parset_path.parent / "updated.parset"
    with updated_parset_path.open("w") as fp:
        parset.write(fp)
    return updated_parset_path


@pytest.mark.parametrize("help_option", ["--help", "-h", None])
@pytest.mark.integration
def test_rapthor_help(help_option):
    """Test the Rapthor pipeline CLI options."""
    command = ["rapthor", help_option] if help_option else ["rapthor"]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage: rapthor <parset>" in result.stdout
    assert "Options:" in result.stdout


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
def test_rapthor_run_single_loop(generated_parset_path, single_loop_strategy_path):
    """Test a single self-calibration loop end to end."""

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {"allow_internet_access": "False", "strategy": str(single_loop_strategy_path)},
    )

    command = ["rapthor", str(updated_parset_path)]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0
    assert "Operation calibrate_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    assert "Rapthor has finished :)" in output
