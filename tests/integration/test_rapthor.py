"""Integration tests for the Rapthor pipeline."""

import configparser
import shlex
import subprocess

import pytest


def update_parset_path(parset_path, param_dict):
    """Helper function to update parset parameters and return a new path."""
    parset = configparser.ConfigParser()
    parset.read(parset_path)
    missing_params = set(param_dict.keys())

    for section in parset.sections():
        for key, value in param_dict.items():
            if key in parset[section]:
                parset[section][key] = value
                missing_params.discard(key)

    updated_parset_path = parset_path.parent / "updated.parset"

    if missing_params:
        raise ValueError(f"Parameters {missing_params} not found in parset.")

    with updated_parset_path.open("w") as fp:
        parset.write(fp)
    return updated_parset_path


def test_update_parset_path(tmp_path):
    """Test the update_parset_path helper function."""
    parset_content = """[section1]
                        param1 = value1
                        param2 = value2
                        [section2]
                        param3 = value3
                        param4 = value4"""
    parset_path = tmp_path / "test.parset"
    parset_path.write_text(parset_content)
    updated_parset_path = update_parset_path(
        parset_path, {"param1": "new_value1", "param3": "new_value3"}
    )
    updated_parset = configparser.ConfigParser()
    updated_parset.read(updated_parset_path)
    assert updated_parset["section1"]["param1"] == "new_value1"
    assert updated_parset["section1"]["param2"] == "value2"
    assert updated_parset["section2"]["param3"] == "new_value3"
    assert updated_parset["section2"]["param4"] == "value4"


def test_update_parset_path_missing_param(tmp_path):
    """Test the update_parset_path helper function with a missing parameter."""
    parset_content = """[section1]
                        param1 = value1
                        param2 = value2
                        [section2]
                        param3 = value3
                        param4 = value4"""
    parset_path = tmp_path / "test.parset"
    parset_path.write_text(parset_content)
    with pytest.raises(ValueError, match="Parameters .* not found in parset."):
        update_parset_path(parset_path, {"param1": "new_value1", "param5": "new_value5"})


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


@pytest.mark.integration
@pytest.mark.parametrize(
    "shared_facet_rw", [True, False], ids=["shared_facet_rw_true", "shared_facet_rw_false"]
)
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
def test_rapthor_run_single_loop(generated_parset_path, single_loop_strategy_path, shared_facet_rw):
    """Test a single self-calibration loop end to end."""

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path),
            "shared_facet_rw": str(shared_facet_rw),
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


def test_multiple_sectors_created(
    generated_parset_path,
    single_loop_strategy_path,
    tmp_path,
):
    """Test that multiple sectors are produced when grid_nsectors_ra > 1."""

    """ --- Run 1: single sector --- """
    parset_single = update_parset_path(
        generated_parset_path,
        {
            "strategy": str(single_loop_strategy_path),
            "allow_internet_access": "False",
            "grid_nsectors_ra": "0",
        },
    )

    run1_dir = tmp_path / "run_single"
    run1_dir.mkdir()

    command_single = ["rapthor", str(parset_single)]
    result1 = subprocess.run(
        command_single,
        cwd=run1_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result1.returncode == 0

    single_outputs = list(run1_dir.rglob("sector_1-*.fits*"))
    single_sector2 = list(run1_dir.rglob("sector_2-*.fits*"))

    assert len(single_outputs) > 0
    assert len(single_sector2) == 0, "Unexpected sector_2 files in single-sector run"

    """ --- Run 2: multiple sectors --- """
    parset_multi = update_parset_path(
        generated_parset_path,
        {
            "strategy": str(single_loop_strategy_path),
            "allow_internet_access": "False",
            "grid_nsectors_ra": "2",
        },
    )

    run2_dir = tmp_path / "run_multi"
    run2_dir.mkdir()

    command_multi = ["rapthor", str(parset_multi)]
    result2 = subprocess.run(
        command_multi,
        cwd=run2_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result2.returncode == 0

    multi_outputs = list(run2_dir.rglob("sector_1-*.fits*"))
    multi_sector2 = list(run2_dir.rglob("sector_2-*.fits*"))

    assert len(multi_outputs) > 0
    assert len(multi_sector2) > 0, "Expected sector_2 files not found in multi-sector run"
