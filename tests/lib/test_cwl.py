"""
Test module for testing the CWL workflows _generated_ by the pipeline
"""

import subprocess
from pathlib import Path

import pytest
from rapthor.lib.operation import DIR, env_parset


def generate_and_validate(tmp_path, operation, parms, templ, sub_templ=None):
    """
    Generate a CWL workflow using `templ`, and sub-workflow if `sub_templ` is given,
    the same way that `rapthor.lib.operation.Operation.setup()` does this.
    Validate the workflow file using `cwltool`.
    """
    if parms.get("use_facets") and parms.get("apply_screens"):
        pytest.skip("'use_facets' and 'apply_screens' cannot both be enabled")
    if parms.get("normalize_flux_scale") and not parms.get("make_image_cube"):
        pytest.skip("'normalize_flux_scale' must be used with 'make_image_cube'")
    if (parms.get("use_facets") or parms.get("apply_screens")) and parms.get("preapply_dde_solutions"):
        pytest.skip("'preapply_dde_solutions' cannot be used with 'use_facets' or 'apply_screens'")
    pipeline_working_dir = tmp_path / "pipelines" / operation
    pipeline_working_dir.mkdir(parents=True, exist_ok=True)
    parset = pipeline_working_dir / "pipeline_parset.cwl"
    sub_parset = pipeline_working_dir / "subpipeline_parset.cwl"
    rapthor_pipeline_dir = (Path(DIR) / "../pipeline").resolve()
    parset.write_text(
        templ.render(
            parms,
            pipeline_working_dir=pipeline_working_dir,
            rapthor_pipeline_dir=rapthor_pipeline_dir,
        )
    )
    if sub_templ:
        sub_parset.write_text(
            sub_templ.render(
                parms,
                pipeline_working_dir=pipeline_working_dir,
                rapthor_pipeline_dir=rapthor_pipeline_dir,
            )
        )
    try:
        subprocess.run(["cwltool", "--validate", "--enable-ext", parset], check=True)
    except subprocess.CalledProcessError as err:
        raise AssertionError(f"FAILED with parameters: {parms}") from err


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
    templ = env_parset.get_template("concatenate_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
    }
    generate_and_validate(tmp_path, operation, parms, templ)


@pytest.mark.parametrize("use_image_based_predict", (False, True))
@pytest.mark.parametrize("do_slowgain_solve", (False, True))
@pytest.mark.parametrize("max_cores", (None, 8))
def test_calibrate_workflow(
    tmp_path,
    use_image_based_predict,
    do_slowgain_solve,
    max_cores,
):
    """
    Test the Calibrate workflow, using all possible combinations of parameters that
    control the way the CWL workflow is generated from the template. Parameters were
    taken from `CalibrateDD.set_parset_parameters()`.
    """
    operation = "calibrate"
    templ = env_parset.get_template("calibrate_pipeline.cwl")
    parms = {
        "use_image_based_predict": use_image_based_predict,
        "do_slowgain_solve": do_slowgain_solve,
        "max_cores": max_cores,
    }
    generate_and_validate(tmp_path, operation, parms, templ)


@pytest.mark.parametrize("max_cores", (None, 8))
def test_calibrate_di_workflow(
    tmp_path,
    max_cores,
):
    """
    Test the Calibrate DI workflow, using all possible combinations of parameters
    that control the way the CWL workflow is generated from the template. Parameters
    were taken from `CalibrateDI.set_parset_parameters()`.
    """
    operation = "calibrate_di"
    templ = env_parset.get_template("calibrate_di_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
    }
    generate_and_validate(tmp_path, operation, parms, templ)


@pytest.mark.parametrize("max_cores", (None, 8))
def test_predict_workflow(tmp_path, max_cores):
    """
    Test the Predict workflow, using all possible combinations of parameters that
    control the way the CWL workflow is generated from the template. Parameters were
    taken from `PredictDD.set_parset_parameters()`.
    """
    operation = "predict"
    templ = env_parset.get_template("predict_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
    }
    generate_and_validate(tmp_path, operation, parms, templ)


@pytest.mark.parametrize("max_cores", (None, 8))
def test_predict_di_workflow(tmp_path, max_cores):
    """
    Test the Predict DI workflow, using all possible combinations of parameters that
    control the way the CWL workflow is generated from the template. Parameters were
    taken from `PredictDI.set_parset_parameters()`.
    """
    operation = "predict_di"
    templ = env_parset.get_template("predict_di_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
    }
    generate_and_validate(tmp_path, operation, parms, templ)


@pytest.mark.parametrize("max_cores", (None, 8))
def test_predict_nc_workflow(tmp_path, max_cores):
    """
    Test the Predict NC workflow, using all possible combinations of parameters that
    control the way the CWL workflow is generated from the template. Parameters were
    taken from `PredictNC.set_parset_parameters()`.
    """
    operation = "predict_nc"
    templ = env_parset.get_template("predict_nc_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
    }
    generate_and_validate(tmp_path, operation, parms, templ)


@pytest.mark.slow
@pytest.mark.parametrize("apply_screens", (False, True))
@pytest.mark.parametrize("use_facets", (False, True))
@pytest.mark.parametrize("peel_bright_sources", (False, True))
@pytest.mark.parametrize("max_cores", (None, 8))
@pytest.mark.parametrize("use_mpi", (False, True))
@pytest.mark.parametrize("make_image_cube", (False, True))
@pytest.mark.parametrize("normalize_flux_scale", (False, True))
@pytest.mark.parametrize("preapply_dde_solutions", (False, True))
@pytest.mark.parametrize("save_source_list", (False, True))
@pytest.mark.parametrize("compress_images", (False, True))
@pytest.mark.parametrize("filter_by_mask", (True, ))
@pytest.mark.parametrize("source_finder", ("bdsf", "sofia"))
def test_image_workflow(
    tmp_path,
    apply_screens,
    use_facets,
    peel_bright_sources,
    max_cores,
    use_mpi,
    make_image_cube,
    normalize_flux_scale,
    preapply_dde_solutions,
    save_source_list,
    compress_images,
    filter_by_mask,
    source_finder
):
    """
    Test the Image workflow, using all possible combinations of parameters that
    control the way the CWL workflow is generated from the template. Parameters were
    taken from `Image.set_parset_parameters()`.
    """
    operation = "image"
    templ = env_parset.get_template("image_pipeline.cwl")
    sub_templ = env_parset.get_template("image_sector_pipeline.cwl")
    parms = {
        "apply_screens": apply_screens,
        "use_facets": use_facets,
        "peel_bright_sources": peel_bright_sources,
        "max_cores": max_cores,
        "use_mpi": use_mpi,
        "make_image_cube": make_image_cube,
        "normalize_flux_scale": normalize_flux_scale,
        "preapply_dde_solutions": preapply_dde_solutions,
        "save_source_list": save_source_list,
        "compress_images": compress_images,
        "filter_by_mask": filter_by_mask,
        "source_finder": source_finder,
    }
    generate_and_validate(tmp_path, operation, parms, templ, sub_templ)


@pytest.mark.parametrize("max_cores", (None, 8))
@pytest.mark.parametrize("skip_processing", (False, True))
@pytest.mark.parametrize("compress_images", (False, True))
def test_mosaic_workflow(tmp_path, max_cores, skip_processing, compress_images):
    """
    Test the Mosaic workflow, using all possible combinations of parameters that
    control the way the CWL workflow is generated from the template. Parameters were
    taken from `Mosaic.set_parset_parameters()`.
    """
    operation = "mosaic"
    templ = env_parset.get_template("mosaic_pipeline.cwl")
    sub_templ = env_parset.get_template("mosaic_type_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
        "skip_processing": skip_processing,
        "compress_images": compress_images,
    }
    generate_and_validate(tmp_path, operation, parms, templ, sub_templ)
