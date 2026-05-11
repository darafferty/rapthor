"""Integration tests for the Rapthor pipeline when using the do_normalize step."""

import subprocess
from pathlib import Path

import pytest

from .utils import get_working_dir_from_parset, update_parset_path


@pytest.mark.internet
@pytest.mark.integration
@pytest.mark.parametrize(
    "generated_parset_path_normalisation",
    [
        (
            "tests/resources/integration_template.parset",
            "tests/resources/integration_true_sky.txt",
            "tests/resources/integration_apparent_sky.txt",
        )
    ],
    indirect=True,
)
def test_rapthor_run_single_loop_with_do_normalize(
    generated_parset_path_normalisation, single_loop_do_normalize_strategy_path
):
    """Test a single self-calibration loop end to end."""

    updated_parset_path = update_parset_path(
        generated_parset_path_normalisation,
        {
            "allow_internet_access": "True",
            "strategy": str(single_loop_do_normalize_strategy_path),
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
    assert "Operation predict_1 completed" in output
    assert "Operation normalize_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    assert "Rapthor has finished :)" in output


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
def test_rapthor_run_single_loop_with_do_normalize_no_internet_raises_error(
    generated_parset_path, single_loop_do_normalize_strategy_path
):
    """Test that rapthor raises an error when do_normalize is used without internet access."""

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_do_normalize_strategy_path),
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
    assert result.returncode != 0, (
        f"Rapthor should have failed but succeeded with output:\n{output}"
    )
    assert (
        "The strategy includes do_normalize in the first cycle, which requires internet access "
        in output
    )


@pytest.mark.internet
@pytest.mark.integration
@pytest.mark.parametrize(
    "generated_parset_path_normalisation",
    [
        (
            "tests/resources/integration_template.parset",
            "tests/resources/integration_true_sky.txt",
            "tests/resources/integration_apparent_sky.txt",
        )
    ],
    indirect=True,
)
def test_rapthor_run_single_loop_with_do_normalize_no_internet_provided_sky_models_ok(
    generated_parset_path_normalisation, single_loop_do_normalize_strategy_path
):
    """Test that rapthor runs successfully when do_normalize is used without internet access but sky models are provided."""

    updated_parset_path = update_parset_path(
        generated_parset_path_normalisation,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_do_normalize_strategy_path),
            "normalization_skymodels": "[tests/resources/integration_apparent_sky.txt, tests/resources/integration_true_sky.txt]",
            "normalization_reference_frequencies": "[150000000.0, 150000000.0]",
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
    assert "Operation predict_1 completed" in output
    assert "Operation normalize_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    assert "Rapthor has finished :)" in output


@pytest.mark.integration
@pytest.mark.parametrize("normalization_skymodel_paths", [None], indirect=True)
@pytest.mark.parametrize(
    "generated_parset_path_normalisation",
    [
        (
            "tests/resources/integration_template.parset",
            "tests/resources/integration_true_sky.txt",
            "tests/resources/integration_apparent_sky.txt",
        )
    ],
    indirect=True,
)
def test_rapthor_run_single_loop_with_do_normalize_no_matching_sources_skips_normalization(
    generated_parset_path_normalisation,
    no_matching_normalization_inputs,
    normalization_skymodel_paths,
):
    """Test do_normalize skips normalization when reference sky models do not match."""
    updated_parset_path = update_parset_path(
        generated_parset_path_normalisation,
        no_matching_normalization_inputs,
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
    assert "Operation normalize_1 completed" in output
    assert "Rapthor has finished :)" in output

    working_dir = get_working_dir_from_parset(updated_parset_path)
    normalize_logs_dir = Path(working_dir) / "logs" / "normalize_1"
    normalize_logs = sorted(normalize_logs_dir.rglob("*normalize_flux_scale*.log"))
    assert normalize_logs, f"No normalize_flux_scale logs found in {normalize_logs_dir}"
    normalize_log_text = "\n".join(log_path.read_text() for log_path in normalize_logs)
    assert "Flux normalization will be skipped" in normalize_log_text
