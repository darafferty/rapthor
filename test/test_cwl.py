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
    if parms.get("use_facets") and parms.get("apply_screens"):
        pytest.skip("'use_facets' and 'apply_screens' cannot both be enabled")
    if parms.get("apply_normalizations") and parms.get("normalize_flux_scale"):
        pytest.skip("'apply_normalizations' and 'normalize_flux_scale' cannot both be enabled")
    if parms.get("preapply_dde_solutions") and (parms.get("use_facets") or parms.get("apply_screens")):
        pytest.skip("'preapply_dde_solutions' cannot be used with facets or screens")
    if parms.get("apply_none") and (parms.get("use_facets") or
                                    parms.get("apply_screens") or
                                    parms.get("preapply_dde_solutions") or
                                    parms.get("apply_amplitudes") or
                                    parms.get("apply_normalizations") or
                                    parms.get("apply_fulljones")):
        pytest.skip("'apply_none' cannot be used with any other apply flags, facets, or screens")
    if parms.get("normalize_flux_scale") and not parms.get("make_image_cube"):
        pytest.skip("'normalize_flux_scale' must be used with 'make_image_cube'")
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


@pytest.mark.parametrize("generate_screens", (False, True))
@pytest.mark.parametrize("use_facets", (False, True))
@pytest.mark.parametrize("do_slowgain_solve", (False, True))
@pytest.mark.parametrize("do_joint_solve", (False, True))
@pytest.mark.parametrize("use_scalarphase", (False, True))
@pytest.mark.parametrize("apply_diagonal_solutions", (False, True))
@pytest.mark.parametrize("max_cores", (None, 8))
def test_calibrate_workflow(
    tmp_path,
    generate_screens,
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
        "generate_screens": generate_screens,
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


@pytest.mark.parametrize("max_cores", (None, 8))
@pytest.mark.parametrize("apply_solutions", (False, True))
@pytest.mark.parametrize("apply_amplitudes", (False, True))
def test_predict_nc_workflow(tmp_path, max_cores, apply_solutions, apply_amplitudes):
    """
    Test the Predict NC workflow, using all possible combinations of parameters that
    control the way the CWL workflow is generated from the template. Parameters were
    taken from `PredictNC.set_parset_parameters()`.
    """
    operation = "predict_nc"
    templ = rapthor.lib.operation.env_parset.get_template("predict_nc_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
        "apply_solutions": apply_solutions,
        "apply_amplitudes": apply_amplitudes,
    }
    generate_and_validate(tmp_path, operation, parms, templ)


@pytest.mark.parametrize("apply_none", (False, True))
@pytest.mark.parametrize("apply_amplitudes", (False, True))
@pytest.mark.parametrize("apply_screens", (False, True))
@pytest.mark.parametrize("apply_normalizations", (False, True))
@pytest.mark.parametrize("apply_fulljones", (False, True))
@pytest.mark.parametrize("preapply_dde_solutions", (False, True))
@pytest.mark.parametrize("use_facets", (False, True))
@pytest.mark.parametrize("peel_bright_sources", (False, True))
@pytest.mark.parametrize("max_cores", (None, 8))
@pytest.mark.parametrize("use_mpi", (False, True))
@pytest.mark.parametrize("make_image_cube", (False, True))
@pytest.mark.parametrize("normalize_flux_scale", (False, True))
def test_image_workflow(
    tmp_path,
    apply_none,
    apply_amplitudes,
    apply_screens,
    apply_normalizations,
    apply_fulljones,
    preapply_dde_solutions,
    use_facets,
    peel_bright_sources,
    max_cores,
    use_mpi,
    make_image_cube,
    normalize_flux_scale
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
        "apply_none": apply_none,
        "apply_amplitudes": apply_amplitudes,
        "apply_screens": apply_screens,
        "apply_normalizations": apply_normalizations,
        "apply_fulljones": apply_fulljones,
        "preapply_dde_solutions": preapply_dde_solutions,
        "use_facets": use_facets,
        "peel_bright_sources": peel_bright_sources,
        "max_cores": max_cores,
        "use_mpi": use_mpi,
        "make_image_cube": make_image_cube,
        "normalize_flux_scale": normalize_flux_scale,
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
