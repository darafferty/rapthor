"""
Test module for testing the CWL workflows _generated_ by the pipeline
"""

import os
import pytest
import subprocess

import rapthor.lib.operation


def generate_and_validate(tmp_path, operation, parms, templ, sub_templ=None):
    """
    Generate a CWL workflow using `templ`, and sub-workflow if `sub_templ` is given,
    the same way that `rapthor.lib.operation.Operation.setup()` does this.
    Validate the workflow file using `cwltool`.
    """
    if parms.get("use_facets") and parms.get("use_screens"):
        pytest.skip("'use_facets' and 'use_screens' cannot be enabled both")
    pipeline_working_dir = tmp_path / "pipelines" / operation
    pipeline_working_dir.mkdir(parents=True, exist_ok=True)
    parset = pipeline_working_dir / "pipeline_parset.cwl"
    sub_parset = pipeline_working_dir / "subpipeline_parset.cwl"
    rapthor_pipeline_dir = os.path.abspath(
        os.path.join(rapthor.lib.operation.DIR, "..", "pipeline")
    )
    with open(parset, "w") as f:
        f.write(
            templ.render(
                parms,
                pipeline_working_dir=pipeline_working_dir,
                rapthor_pipeline_dir=rapthor_pipeline_dir,
            )
        )
    if sub_templ:
        with open(sub_parset, "w") as f:
            f.write(
                sub_templ.render(
                    parms,
                    pipeline_working_dir=pipeline_working_dir,
                    rapthor_pipeline_dir=rapthor_pipeline_dir,
                )
            )
    try:
        subprocess.run(["cwltool", "--validate", "--enable-ext", parset], check=True)
    except subprocess.CalledProcessError as err:
        raise Exception(f"FAILED with parameters: {parms}")


@pytest.mark.parametrize("max_cores", (None, 8))
def test_concatenate_workflow(
    tmp_path,
    max_cores,
):
    """
    Test the Concatenate workflow, using all possible combinations of parameters that
    control the way the CWL workflow is generated from the template. Parameters were
    taken from `Concatenate.set_parset_parameters()`.
    """
    operation = "concatenate"
    templ = rapthor.lib.operation.env_parset.get_template("concatenate_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
    }
    generate_and_validate(tmp_path, operation, parms, templ)


@pytest.mark.parametrize("use_screens", (False, True))
@pytest.mark.parametrize("use_facets", (False, True))
@pytest.mark.parametrize("do_slowgain_solve", (False, True))
@pytest.mark.parametrize("do_joint_solve", (False, True))
@pytest.mark.parametrize("use_scalarphase", (False, True))
@pytest.mark.parametrize("apply_diagonal_solutions", (False, True))
@pytest.mark.parametrize("max_cores", (None, 8))
def test_calibrate_workflow(
    tmp_path,
    use_screens,
    use_facets,
    do_slowgain_solve,
    do_joint_solve,
    use_scalarphase,
    apply_diagonal_solutions,
    max_cores,
):
    """
    Test the Calibrate workflow, using all possible combinations of parameters that
    control the way the CWL workflow is generated from the template. Parameters were
    taken from `CalibrateDD.set_parset_parameters()`.
    """
    operation = "calibrate"
    templ = rapthor.lib.operation.env_parset.get_template("calibrate_pipeline.cwl")
    parms = {
        "use_screens": use_screens,
        "use_facets": use_facets,
        "do_slowgain_solve": do_slowgain_solve,
        "do_joint_solve": do_joint_solve,
        "use_scalarphase": use_scalarphase,
        "apply_diagonal_solutions": apply_diagonal_solutions,
        "max_cores": max_cores,
    }
    generate_and_validate(tmp_path, operation, parms, templ)


@pytest.mark.parametrize("do_fulljones_solve", (False, True))
@pytest.mark.parametrize("max_cores", (None, 8))
def test_calibrate_di_workflow(
    tmp_path,
    do_fulljones_solve,
    max_cores,
):
    """
    Test the Calibrate DI workflow, using all possible combinations of parameters
    that control the way the CWL workflow is generated from the template. Parameters
    were taken from `CalibrateDI.set_parset_parameters()`.
    """
    operation = "calibrate_di"
    templ = rapthor.lib.operation.env_parset.get_template("calibrate_di_pipeline.cwl")
    parms = {
        "do_fulljones_solve": do_fulljones_solve,
        "max_cores": max_cores,
    }
    generate_and_validate(tmp_path, operation, parms, templ)


@pytest.mark.parametrize("max_cores", (None, 8))
@pytest.mark.parametrize("apply_amplitudes", (False, True))
def test_predict_workflow(tmp_path, max_cores, apply_amplitudes):
    """
    Test the Predict workflow, using all possible combinations of parameters that
    control the way the CWL workflow is generated from the template. Parameters were
    taken from `PredictDD.set_parset_parameters()`.
    """
    operation = "predict"
    templ = rapthor.lib.operation.env_parset.get_template("predict_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
        "apply_amplitudes": apply_amplitudes,
    }
    generate_and_validate(tmp_path, operation, parms, templ)


@pytest.mark.parametrize("max_cores", (None, 8))
@pytest.mark.parametrize("apply_amplitudes", (False, True))
def test_predict_di_workflow(tmp_path, max_cores, apply_amplitudes):
    """
    Test the Predict DI workflow, using all possible combinations of parameters that
    control the way the CWL workflow is generated from the template. Parameters were
    taken from `PredictDI.set_parset_parameters()`.
    """
    operation = "predict_di"
    templ = rapthor.lib.operation.env_parset.get_template("predict_di_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
        "apply_amplitudes": apply_amplitudes,
    }
    generate_and_validate(tmp_path, operation, parms, templ)


@pytest.mark.parametrize("apply_amplitudes", (False, True))
@pytest.mark.parametrize("use_screens", (False, True))
@pytest.mark.parametrize("use_facets", (False, True))
@pytest.mark.parametrize("peel_bright_sources", (False, True))
@pytest.mark.parametrize("max_cores", (None, 8))
@pytest.mark.parametrize("use_mpi", (False, True))
def test_image_workflow(
    tmp_path,
    apply_amplitudes,
    use_screens,
    use_facets,
    peel_bright_sources,
    max_cores,
    use_mpi,
):
    """
    Test the Image workflow, using all possible combinations of parameters that
    control the way the CWL workflow is generated from the template. Parameters were
    taken from `Image.set_parset_parameters()`.
    """
    operation = "image"
    templ = rapthor.lib.operation.env_parset.get_template("image_pipeline.cwl")
    sub_templ = rapthor.lib.operation.env_parset.get_template(
        "image_sector_pipeline.cwl"
    )
    parms = {
        "apply_amplitudes": apply_amplitudes,
        "use_screens": use_screens,
        "use_facets": use_facets,
        "peel_bright_sources": peel_bright_sources,
        "max_cores": max_cores,
        "use_mpi": use_mpi,
    }
    generate_and_validate(tmp_path, operation, parms, templ, sub_templ)


@pytest.mark.parametrize("max_cores", (None, 8))
@pytest.mark.parametrize("skip_processing", (False, True))
def test_mosaic_workflow(tmp_path, max_cores, skip_processing):
    """
    Test the Mosaic workflow, using all possible combinations of parameters that
    control the way the CWL workflow is generated from the template. Parameters were
    taken from `Mosaic.set_parset_parameters()`.
    """
    operation = "mosaic"
    templ = rapthor.lib.operation.env_parset.get_template("mosaic_pipeline.cwl")
    sub_templ = rapthor.lib.operation.env_parset.get_template(
        "mosaic_type_pipeline.cwl"
    )
    parms = {
        "max_cores": max_cores,
        "skip_processing": skip_processing,
    }
    generate_and_validate(tmp_path, operation, parms, templ, sub_templ)
