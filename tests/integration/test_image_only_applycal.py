import json
import subprocess
from pathlib import Path

import pytest

from .utils import (
    find_step_logs,
    get_working_dir_from_parset,
    parse_dp3_args_from_log,
    update_parset_path,
)

INTEGRATION_PARSET = (
    "tests/resources/integration_template.parset",
    "tests/resources/integration_true_sky.txt",
    "tests/resources/integration_apparent_sky.txt",
)


def _pipeline_log_tails(parset_path, line_count=100):
    """Return the end of each pipeline log for a failed Rapthor run."""
    working_dir = Path(get_working_dir_from_parset(parset_path))
    sections = []
    for log_path in sorted((working_dir / "logs").glob("*/pipeline.log")):
        lines = log_path.read_text(errors="replace").splitlines()
        sections.append(f"--- {log_path} ---\n" + "\n".join(lines[-line_count:]))
    return "\n".join(sections)


def _run_rapthor(parset_path):
    command = ["rapthor", str(parset_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = f"{result.stdout}\n{result.stderr}"
    pipeline_logs = _pipeline_log_tails(parset_path) if result.returncode else ""
    assert result.returncode == 0, (
        f"Rapthor failed with output:\n{output}\nPipeline log tails:\n{pipeline_logs}"
    )
    return output


def _image_pipeline_inputs(working_dir, image_name):
    inputs_path = Path(working_dir) / "pipelines" / image_name / "pipeline_inputs.json"
    assert inputs_path.exists(), f"Expected image pipeline inputs at {inputs_path}"
    return json.loads(inputs_path.read_text())


def _prepare_imaging_data_args(working_dir, image_name):
    image_logs_dir = Path(working_dir) / "logs" / image_name
    prepare_logs = find_step_logs(image_logs_dir, "prepare_imaging_data.cwl")
    assert prepare_logs, "Expected prepare_imaging_data logs to be present"
    return parse_dp3_args_from_log(prepare_logs[0])


@pytest.mark.integration
@pytest.mark.parametrize("generated_parset_path", [INTEGRATION_PARSET], indirect=True)
def test_image_only_run_applies_supplied_input_h5parm(
    generated_parset_path,
    image_only_strategy_path,
    resource_dir,
):
    """Image-only runs use the parset-supplied H5parm to build applycal."""
    input_h5parm = resource_dir / "integration_field_solutions.h5"
    image_working_dir = Path(get_working_dir_from_parset(generated_parset_path))
    image_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(image_only_strategy_path),
            "input_h5parm": str(input_h5parm),
            "dde_method": "single",
            "reweight": "False",
            "regroup_input_skymodel": "False",
        },
    )
    print("---Rapthor working dir: ", image_working_dir)

    image_output = _run_rapthor(image_parset_path)
    assert "Operation calibrate_1 completed" not in image_output
    assert "Operation image_1 completed" in image_output
    assert "Rapthor has finished :)" in image_output

    image_inputs = _image_pipeline_inputs(image_working_dir, "image_1")
    assert image_inputs["prepare_data_applycal_steps"] == "[fastphase]"
    assert "applycal" in image_inputs["prepare_data_steps"]
    assert image_inputs["prepare_data_h5parm"]["path"] == str(input_h5parm)
    assert image_inputs["h5parm"]["path"] == str(input_h5parm)

    dp3_arguments = _prepare_imaging_data_args(image_working_dir, "image_1")
    assert "applycal" in dp3_arguments["steps"]
    assert dp3_arguments["applycal.steps"] == "[fastphase]"
    assert Path(dp3_arguments["applycal.parmdb"]).name == input_h5parm.name


@pytest.mark.integration
@pytest.mark.parametrize("generated_parset_path", [INTEGRATION_PARSET], indirect=True)
def test_image_only_final_cycle_applies_previous_cycle_h5parm(
    generated_parset_path,
    calibrate_then_image_only_strategy_path,
):
    """A later image-only cycle applies the solution made in the selfcal cycle."""
    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(calibrate_then_image_only_strategy_path),
            "dde_method": "single",
            "reweight": "False",
        },
    )
    working_dir = Path(get_working_dir_from_parset(updated_parset_path))
    print("---Rapthor working dir: ", working_dir)

    output = _run_rapthor(updated_parset_path)
    assert "Operation calibrate_di_1 completed" in output
    assert "Operation calibrate_1 completed" in output
    assert "Operation image_1 completed" in output
    assert "Operation image_2 completed" in output
    assert "Rapthor has finished :)" in output

    previous_h5parm = working_dir / "solutions" / "calibrate_1" / "field-solutions.h5"
    previous_fulljones_h5parm = (
        working_dir / "solutions" / "calibrate_di_1" / "fulljones-solutions.h5"
    )
    assert previous_h5parm.exists(), "Expected the selfcal cycle to produce scalar solutions"
    assert previous_fulljones_h5parm.exists(), "Expected the selfcal cycle to produce full-Jones"

    image_inputs = _image_pipeline_inputs(working_dir, "image_2")
    assert image_inputs["prepare_data_applycal_steps"] == "[fastphase,fulljones]"
    assert "applycal" in image_inputs["prepare_data_steps"]
    assert image_inputs["prepare_data_h5parm"]["path"] == str(previous_h5parm)
    assert image_inputs["h5parm"]["path"] == str(previous_h5parm)
    assert image_inputs["fulljones_h5parm"]["path"] == str(previous_fulljones_h5parm)

    dp3_arguments = _prepare_imaging_data_args(working_dir, "image_2")
    assert "applycal" in dp3_arguments["steps"]
    assert dp3_arguments["applycal.steps"] == "[fastphase,fulljones]"
    assert Path(dp3_arguments["applycal.parmdb"]).name == previous_h5parm.name
    assert Path(dp3_arguments["applycal.fulljones.parmdb"]).name == previous_fulljones_h5parm.name


@pytest.mark.integration
@pytest.mark.parametrize("generated_parset_path", [INTEGRATION_PARSET], indirect=True)
def test_image_only_final_cycle_preapplies_di_slow_and_images_with_dd(
    generated_parset_path,
    calibrate_di_slow_dd_fast_then_image_only_strategy_path,
):
    """Later image-only facet cycles preapply DI diagonal gains and image with DD gains."""
    updated_parset_path = update_parset_path(
        generated_parset_path,
        {
            "allow_internet_access": "False",
            "strategy": str(calibrate_di_slow_dd_fast_then_image_only_strategy_path),
            "dde_method": "full",
            "reweight": "False",
        },
    )
    working_dir = Path(get_working_dir_from_parset(updated_parset_path))
    print("---Rapthor working dir: ", working_dir)

    output = _run_rapthor(updated_parset_path)
    assert "Operation calibrate_di_1 completed" in output
    assert "Operation calibrate_1 completed" in output
    assert "Operation image_2 completed" in output
    assert "Rapthor has finished :)" in output

    previous_di_h5parm = working_dir / "solutions" / "calibrate_di_1" / "di-solutions.h5"
    previous_dd_h5parm = working_dir / "solutions" / "calibrate_1" / "field-solutions.h5"
    assert previous_di_h5parm.exists(), "Expected the selfcal cycle to produce DI solutions"
    assert previous_dd_h5parm.exists(), "Expected the selfcal cycle to produce DD solutions"

    image_inputs = _image_pipeline_inputs(working_dir, "image_2")
    assert image_inputs["prepare_data_applycal_steps"] == "[slowgain]"
    assert "applycal" in image_inputs["prepare_data_steps"]
    assert image_inputs["prepare_data_h5parm"]["path"] == str(previous_di_h5parm)
    assert image_inputs["h5parm"]["path"] == str(previous_dd_h5parm)

    dp3_arguments = _prepare_imaging_data_args(working_dir, "image_2")
    assert "applycal" in dp3_arguments["steps"]
    assert dp3_arguments["applycal.steps"] == "[slowgain]"
    assert Path(dp3_arguments["applycal.parmdb"]).name == previous_di_h5parm.name
