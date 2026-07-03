"""Focused integration scenarios that protect migration-sensitive workflows."""

import configparser
import json
import shutil
import subprocess
from pathlib import Path

import casacore.tables as pt
import pytest

from .utils import (
    find_command_records,
    first_command_arguments,
    get_working_dir_from_parset,
    update_parset_path,
)


def _run_rapthor(parset_path):
    result = subprocess.run(
        ["rapthor", str(parset_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"Rapthor failed with output:\n{output}"
    assert "Rapthor has finished :)" in output
    return output


def _read_parset(parset_path):
    parset = configparser.ConfigParser()
    parset.read(parset_path)
    return parset


def _copy_ms_with_shifted_frequency(source_ms, target_ms):
    shutil.copytree(source_ms, target_ms)
    with pt.table(f"{target_ms}::SPECTRAL_WINDOW", readonly=False, ack=False) as table:
        channel_freq = table.getcol("CHAN_FREQ")
        channel_width = table.getcol("CHAN_WIDTH")
        offset_hz = float(channel_freq.max() - channel_freq.min() + 2 * abs(channel_width).max())
        table.putcol("CHAN_FREQ", channel_freq + offset_hz)
        table.putcol("REF_FREQUENCY", table.getcol("REF_FREQUENCY") + offset_hz)
    return target_ms


def _image_pipeline_inputs(working_dir, image_name):
    inputs_path = Path(working_dir) / "pipelines" / image_name / "pipeline_inputs.json"
    assert inputs_path.exists(), f"Expected image pipeline inputs at {inputs_path}"
    return json.loads(inputs_path.read_text(encoding="utf-8"))


def _assert_wsclean_applies_h5parm(working_dir, operation, h5parm, *, diagonal=False):
    wsclean_commands = find_command_records(
        working_dir,
        operation=operation,
        executable="wsclean",
        contains="-apply-facet-solutions",
    )
    assert wsclean_commands
    assert any(str(h5parm) in record.command for record in wsclean_commands)
    if diagonal:
        assert any("-diagonal-visibilities" in record.command for record in wsclean_commands)


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
def test_rapthor_run_concatenates_multiple_measurement_sets(
    generated_parset_path,
    single_loop_strategy_path,
    tmp_path,
):
    """Run a small reduction where an epoch contains multiple Measurement Sets."""
    generated_parset = _read_parset(generated_parset_path)
    input_ms = Path(generated_parset["global"]["input_ms"])
    shifted_input_ms = _copy_ms_with_shifted_frequency(
        input_ms, tmp_path / "test_shifted_frequency.ms"
    )
    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path),
            "input_ms": f"[{input_ms}, {shifted_input_ms}]",
        },
    )

    output = _run_rapthor(updated_parset_path)

    assert "Operation concatenate_1 completed" in output
    assert "Operation image_1 completed" in output
    working_dir = get_working_dir_from_parset(updated_parset_path)
    concat_commands = find_command_records(
        working_dir,
        operation="concatenate_1",
        executable="DP3",
    )
    assert concat_commands


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
def test_rapthor_run_image_only_with_supplied_h5parm_and_skymodel(
    generated_parset_path,
    calibrate_only_strategy_path,
    image_only_strategy_path,
    tmp_path,
):
    """Produce real solutions, then run an image-only pass with supplied inputs."""
    producer_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(calibrate_only_strategy_path),
        },
    )
    _run_rapthor(producer_parset_path)

    producer_working_dir = Path(get_working_dir_from_parset(producer_parset_path))
    h5parm = producer_working_dir / "solutions" / "calibrate_1" / "field-solutions.h5"
    assert h5parm.is_file(), f"No calibration h5parm was produced in {producer_working_dir}"
    skymodel = producer_working_dir / "skymodels" / "predict_1" / "predict_1_predict_skymodel.txt"
    assert skymodel.is_file(), f"No matching sky model was produced in {producer_working_dir}"

    image_only_working_dir = producer_working_dir.parent / "image-only-work"
    image_only_working_dir.mkdir()
    image_only_parset_path = update_parset_path(
        generated_parset_path,
        {
            "dir_working": str(image_only_working_dir),
            "allow_internet_access": "False",
            "strategy": str(image_only_strategy_path),
            "input_h5parm": str(h5parm),
            "input_skymodel": str(skymodel),
            "apparent_skymodel": "None",
            "regroup_input_skymodel": "False",
        },
    )

    output = _run_rapthor(image_only_parset_path)

    assert "Operation calibrate_1 completed" not in output
    assert "Operation calibrate_di_1 completed" not in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    image_inputs = _image_pipeline_inputs(image_only_working_dir, "image_1")
    assert image_inputs["prepare_data_applycal_steps"] is None
    assert "applycal" not in image_inputs["prepare_data_steps"]
    assert image_inputs["h5parm"]["path"] == str(h5parm)

    _assert_wsclean_applies_h5parm(image_only_working_dir, "image_1", h5parm)


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
def test_rapthor_run_final_image_only_applies_previous_cycle_h5parm(
    generated_parset_path,
    calibrate_then_image_only_strategy_path,
):
    """Run selfcal, then verify the final image-only pass applies its solutions."""
    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(calibrate_then_image_only_strategy_path),
            "reweight": "False",
        },
    )

    output = _run_rapthor(updated_parset_path)

    assert "Operation calibrate_di_1 completed" in output
    assert "Operation calibrate_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation image_2 completed" in output
    assert "Operation mosaic_2 completed" in output

    working_dir = Path(get_working_dir_from_parset(updated_parset_path))
    dd_h5parm = working_dir / "solutions" / "calibrate_1" / "field-solutions.h5"
    fulljones_h5parm = working_dir / "solutions" / "calibrate_di_1" / "fulljones-solutions.h5"
    assert dd_h5parm.is_file(), f"No DD calibration h5parm was produced in {working_dir}"
    assert fulljones_h5parm.is_file(), f"No full-Jones h5parm was produced in {working_dir}"

    image_inputs = _image_pipeline_inputs(working_dir, "image_2")
    assert image_inputs["prepare_data_applycal_steps"] == "[fulljones]"
    assert "applycal" in image_inputs["prepare_data_steps"]
    assert image_inputs["prepare_data_h5parm"] is None
    assert image_inputs["h5parm"]["path"] == str(dd_h5parm)
    assert image_inputs["fulljones_h5parm"]["path"] == str(fulljones_h5parm)

    dp3_arguments = first_command_arguments(
        working_dir,
        operation="image_2",
        executable="DP3",
        contains="applycal",
    )
    assert "applycal" in dp3_arguments["steps"]
    assert dp3_arguments["applycal.steps"] == "[fulljones]"
    assert "applycal.parmdb" not in dp3_arguments
    assert Path(dp3_arguments["applycal.fulljones.parmdb"]).resolve() == fulljones_h5parm.resolve()
    _assert_wsclean_applies_h5parm(working_dir, "image_2", dd_h5parm)


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
def test_rapthor_run_final_image_only_applies_dd_slow_gains_on_the_fly(
    generated_parset_path,
    dd_slow_then_image_only_strategy_path,
):
    """DD slow diagonal gains are applied by WSClean facets, not DP3 pre-apply."""
    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(dd_slow_then_image_only_strategy_path),
            "reweight": "False",
        },
    )

    output = _run_rapthor(updated_parset_path)

    assert "Operation calibrate_1 completed" in output
    assert "Operation image_2 completed" in output
    working_dir = Path(get_working_dir_from_parset(updated_parset_path))
    dd_h5parm = working_dir / "solutions" / "calibrate_1" / "field-solutions.h5"
    assert dd_h5parm.is_file(), f"No DD slow-gain h5parm was produced in {working_dir}"

    image_inputs = _image_pipeline_inputs(working_dir, "image_2")
    assert image_inputs["prepare_data_applycal_steps"] is None
    assert "applycal" not in image_inputs["prepare_data_steps"]
    assert image_inputs["prepare_data_h5parm"] is None
    assert image_inputs["h5parm"]["path"] == str(dd_h5parm)

    _assert_wsclean_applies_h5parm(working_dir, "image_2", dd_h5parm, diagonal=True)


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
def test_rapthor_run_final_image_only_preapplies_di_slow_and_images_with_dd(
    generated_parset_path,
    di_slow_dd_fast_then_image_only_strategy_path,
):
    """DI slow gains are pre-applied before WSClean applies DD gains on the fly."""
    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(di_slow_dd_fast_then_image_only_strategy_path),
            "reweight": "False",
        },
    )

    output = _run_rapthor(updated_parset_path)

    assert "Operation calibrate_di_1 completed" in output
    assert "Operation calibrate_1 completed" in output
    assert "Operation image_2 completed" in output
    working_dir = Path(get_working_dir_from_parset(updated_parset_path))
    di_h5parm = working_dir / "solutions" / "calibrate_di_1" / "di-solutions.h5"
    dd_h5parm = working_dir / "solutions" / "calibrate_1" / "field-solutions.h5"
    assert di_h5parm.is_file(), f"No DI slow-gain h5parm was produced in {working_dir}"
    assert dd_h5parm.is_file(), f"No DD calibration h5parm was produced in {working_dir}"

    image_inputs = _image_pipeline_inputs(working_dir, "image_2")
    assert image_inputs["prepare_data_applycal_steps"] == "[slowgain]"
    assert "applycal" in image_inputs["prepare_data_steps"]
    assert image_inputs["prepare_data_h5parm"]["path"] == str(di_h5parm)
    assert image_inputs["h5parm"]["path"] == str(dd_h5parm)

    dp3_arguments = first_command_arguments(
        working_dir,
        operation="image_2",
        executable="DP3",
        contains="applycal",
    )
    assert "applycal" in dp3_arguments["steps"]
    assert dp3_arguments["applycal.steps"] == "[slowgain]"
    assert Path(dp3_arguments["applycal.parmdb"]).resolve() == di_h5parm.resolve()
    _assert_wsclean_applies_h5parm(working_dir, "image_2", dd_h5parm)


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
def test_rapthor_run_quv_clean_disabled_image_cubes(
    generated_parset_path,
    single_loop_strategy_path,
):
    """Run final full-Stokes imaging with clean disabled and image cubes enabled."""
    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(single_loop_strategy_path),
            "make_quv_images": "True",
            "disable_iquv_clean": "True",
            "save_image_cube": "True",
            "image_cube_stokes_list": "[I, Q, U, V]",
        },
    )

    output = _run_rapthor(updated_parset_path)

    assert "Stokes I, Q, U, and V images will be made" in output
    assert "Operation image_1 completed" in output
    assert "Operation mosaic_1 completed" in output
    working_dir = Path(get_working_dir_from_parset(updated_parset_path))
    assert sorted((working_dir / "images" / "image_1").glob("*_freq_cube.fits"))
    wsclean_commands = find_command_records(
        working_dir,
        operation="image_1",
        executable="wsclean",
        contains="-pol IQUV",
    )
    assert wsclean_commands
    assert any("-niter 0" in record.command_string for record in wsclean_commands)
