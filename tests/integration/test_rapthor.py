"""Integration tests for the Rapthor pipeline."""

import configparser
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from .utils import find_step_logs, get_working_dir_from_parset


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


def get_wsclean_output_mtimes(image_pipeline_dir):
    """Return a mapping of WSClean output product filenames to their modification timestamps"""
    products = {}
    for pattern in [
        "*-MFS-image.fits",
        "*-MFS-image-pb.fits",
        "*-MFS-residual.fits",
        "*-MFS-dirty.fits",
    ]:
        for path in Path(image_pipeline_dir).glob(pattern):
            products[path.name] = path.stat().st_mtime_ns
    return products


def make_failing_filter_skymodel(fake_bin_dir):
    """Create a PATH-injected wrapper for filter_skymodel.py."""
    fake_script = fake_bin_dir / "filter_skymodel.py"
    fake_script.write_text("#!/usr/bin/env python3\nraise SystemExit(1)")
    fake_script.chmod(0o755)
    return fake_script


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
    assert result.returncode == 0
    assert "Operation calibrate_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output

    # Verify bright-source peeling actually ran by checking wsclean_restore step logs.
    working_dir = get_working_dir_from_parset(updated_parset_path)
    image_logs_dir = Path(working_dir) / "logs" / "image_1"
    restore_pb_logs = find_step_logs(image_logs_dir, "restore_pb.wsclean_restore.cwl")
    restore_nonpb_logs = find_step_logs(image_logs_dir, "restore_nonpb.wsclean_restore.cwl")
    restore_pb_stdout_logs = sorted(
        image_logs_dir.glob("subpipeline_parset.cwl.restore_pb.wsclean_restore.cwl.stdout_*.log")
    )
    restore_nonpb_stdout_logs = sorted(
        image_logs_dir.glob("subpipeline_parset.cwl.restore_nonpb.wsclean_restore.cwl.stdout_*.log")
    )

    if peel_bright_sources:
        assert restore_pb_logs, f"No restore_pb wsclean_restore logs found in {image_logs_dir}"
        assert restore_nonpb_logs, (
            f"No restore_nonpb wsclean_restore logs found in {image_logs_dir}"
        )
        assert restore_pb_stdout_logs, f"No restore_pb stdout logs found in {image_logs_dir}"
        assert restore_nonpb_stdout_logs, f"No restore_nonpb stdout logs found in {image_logs_dir}"
        assert "completed success" in restore_pb_logs[-1].read_text()
        assert "completed success" in restore_nonpb_logs[-1].read_text()
        assert "wsclean" in restore_pb_logs[-1].read_text()
        assert "wsclean" in restore_nonpb_logs[-1].read_text()
    else:
        # With when: $(inputs.peel_bright_sources), Toil still emits CWLJob logs,
        # but command stdout logs are not created and there is no "completed success".
        assert not restore_pb_stdout_logs, (
            "restore_pb stdout log should not exist when peeling is disabled"
        )
        assert not restore_nonpb_stdout_logs, (
            "restore_nonpb stdout log should not exist when peeling is disabled"
        )
        assert "completed success" not in restore_pb_logs[-1].read_text(), (
            "restore_pb wsclean_restore should have been skipped when peeling is disabled"
        )
        assert "completed success" not in restore_nonpb_logs[-1].read_text(), (
            "restore_nonpb wsclean_restore should have been skipped when peeling is disabled"
        )

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
def test_rapthor_run_single_loop_calibrate_di(
    generated_parset_path, single_loop_strategy_path_calibrate_di
):
    """Test a single self-calibration loop with DI calibration end to end."""

    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path_calibrate_di),
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
    assert "Operation calibrate_di_1 completed" in output
    assert "Operation predict_1 completed" in output
    assert "Operation predict_di_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    assert "Rapthor has finished :)" in output


@pytest.fixture()
def injected_failing_filterskymodel_env(tmp_path):
    fake_bin_dir = tmp_path / "fake_bin"
    fake_bin_dir.mkdir()
    make_failing_filter_skymodel(fake_bin_dir)

    modified_env = os.environ.copy()
    modified_env["PATH"] = f"{fake_bin_dir}:{modified_env['PATH']}"
    return modified_env


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
def test_rapthor_restart_after_filter_failure_skips_wsclean(
    generated_parset_path,
    single_loop_strategy_path,
    injected_failing_filterskymodel_env,
):
    """Verify that after a filter_skymodel failure, restarting Rapthor does not rerun WSClean and that previous WSClean outputs are reused, not regenerated."""

    failing_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path),
        },
    )

    first_result = subprocess.run(
        ["rapthor", str(failing_parset_path)],
        capture_output=True,
        text=True,
        check=False,
        env=injected_failing_filterskymodel_env,
    )
    first_output = f"{first_result.stdout}\n{first_result.stderr}"
    assert first_result.returncode != 0, (
        "First run should fail in filter_skymodel via injected wrapper script"
    )

    working_dir = get_working_dir_from_parset(failing_parset_path)
    image_pipeline_dir = Path(working_dir) / "pipelines" / "image_1"
    wsclean_outputs_after_first_run = get_wsclean_output_mtimes(image_pipeline_dir)
    assert wsclean_outputs_after_first_run, (
        "Expected WSClean output products before restart, but none were found. "
        f"First run output was:\n{first_output}"
    )

    # Use now the default environment which as the working filter_skymodel
    second_result = subprocess.run(
        ["rapthor", str(failing_parset_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    second_output = f"{second_result.stdout}\n{second_result.stderr}"
    assert second_result.returncode == 0, f"Second run should succeed. Output was:\n{second_output}"
    assert "Rapthor has finished :)" in second_output

    wsclean_outputs_after_second_run = get_wsclean_output_mtimes(image_pipeline_dir)
    assert set(wsclean_outputs_after_second_run) == set(wsclean_outputs_after_first_run), (
        "WSClean output product set changed after restart. "
        f"Before: {sorted(wsclean_outputs_after_first_run)}; "
        f"after: {sorted(wsclean_outputs_after_second_run)}"
    )
    assert wsclean_outputs_after_second_run == wsclean_outputs_after_first_run, (
        "WSClean output product timestamps changed after restart, suggesting WSClean reran."
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
