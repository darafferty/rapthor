"""
Test module for testing the CWL workflows _generated_ by the pipeline
"""

import subprocess
import itertools as itt
from pathlib import Path

import pytest
from rapthor.lib.operation import DIR, env_parset


PIPELINE_PATH = Path(DIR, "..", "pipeline").resolve()


def generate_keyword_combinations(params_pool):
    for values in itt.product(*params_pool.values()):
        if allow_combination(params := dict(zip(params_pool, values))):
            yield params

    # Add a single test case using sofia as source finder
    yield {
        "apply_screens": False,
        "use_facets": True,
        "peel_bright_sources": True,
        "max_cores": None,
        "use_mpi": False,
        "make_image_cube": True,
        "image_cube_stokes_list": "I",
        "normalize_flux_scale": True,
        "preapply_dde_solutions": False,
        "save_source_list": True,
        "compress_images": False,
        "filter_by_mask": True,
        "source_finder": "sofia"
    }


def allow_combination(params):
    if params.get("use_facets") and params.get("apply_screens"):
        # 'use_facets' and 'apply_screens' cannot both be enabled
        return False

    if params.get("normalize_flux_scale") and not params.get("make_image_cube"):
        # 'normalize_flux_scale' must be used with 'make_image_cube'
        return False

    if (params.get("use_facets") or params.get("apply_screens")) and params.get("preapply_dde_solutions"):
        # 'preapply_dde_solutions' cannot be used with 'use_facets' or 'apply_screens'
        return False

    return True


def generate_and_validate(tmp_path, operation, params, template, sub_template=None):

    pipeline_working_dir = tmp_path / "pipelines" / operation
    parset_path = create_parsets(pipeline_working_dir, params, template, sub_template)

    validate(params, parset_path)


def create_parsets(pipeline_working_dir, params, template, sub_template):

    pipeline_working_dir.mkdir(parents=True, exist_ok=True)

    parset_path = pipeline_working_dir / "pipeline_parset.cwl"
    write_parset(template, params, pipeline_working_dir, parset_path)

    if sub_template:
        sub_parset_path = pipeline_working_dir / "subpipeline_parset.cwl"
        write_parset(sub_template, params, pipeline_working_dir, sub_parset_path)

    return parset_path


def write_parset(template, params, pipeline_working_dir, output_path):

    output_path.write_text(
        template.render(
            params,
            pipeline_working_dir=pipeline_working_dir,
            rapthor_pipeline_dir=PIPELINE_PATH,
        )
    )


def validate(params, parset_path):
    """
    Validate the workflow file using `cwltool` in a subprocess call.
    """
    try:
        subprocess.run(["cwltool", "--validate", "--enable-ext", parset_path], check=True)
    except subprocess.CalledProcessError as err:
        raise AssertionError(f"FAILED with parameters: {params}") from err


@pytest.mark.parametrize("max_cores", (None, 8))
def test_concatenate_workflow(
    tmp_path,
    max_cores,
):
    """
    Test the Concatenate workflow, using all possible combinations of
    parameters that control the way the CWL workflow is generated from the
    template. Parameters were taken from `Concatenate.set_parset_parameters()`.
    """
    operation = "concatenate"
    templ = env_parset.get_template("concatenate_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
    }
    generate_and_validate(tmp_path, operation, parms, templ)


@pytest.mark.parametrize("use_image_based_predict", (False, True))
@pytest.mark.parametrize("do_slowgain_solve", (False, True))
@pytest.mark.parametrize("generate_screens", (False, True))
@pytest.mark.parametrize("max_cores", (None, 8))
def test_calibrate_workflow(
    tmp_path,
    use_image_based_predict,
    do_slowgain_solve,
    generate_screens,
    max_cores,
):
    """
    Test the Calibrate workflow, using all possible combinations of parameters
    that control the way the CWL workflow is generated from the template.
    Parameters were taken from `CalibrateDD.set_parset_parameters()`.
    """
    operation = "calibrate"
    templ = env_parset.get_template("calibrate_pipeline.cwl")
    parms = {
        "use_image_based_predict": use_image_based_predict or generate_screens,
        "do_slowgain_solve": do_slowgain_solve,
        "generate_screens": generate_screens,
        "max_cores": max_cores,
    }
    generate_and_validate(tmp_path, operation, parms, templ)


@pytest.mark.parametrize("max_cores", (None, 8))
def test_calibrate_di_workflow(
    tmp_path,
    max_cores,
):
    """
    Test the Calibrate DI workflow, using all possible combinations of
    parameters that control the way the CWL workflow is generated from the
    template. Parameters were taken from `CalibrateDI.set_parset_parameters()`.
    """
    operation = "calibrate_di"
    template = env_parset.get_template("calibrate_di_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
    }
    generate_and_validate(tmp_path, operation, parms, template)


@pytest.mark.parametrize("max_cores", (None, 8))
def test_predict_workflow(tmp_path, max_cores):
    """
    Test the Predict workflow, using all possible combinations of parameters
    that control the way the CWL workflow is generated from the template.
    Parameters were taken from `PredictDD.set_parset_parameters()`.
    """
    operation = "predict"
    template = env_parset.get_template("predict_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
    }
    generate_and_validate(tmp_path, operation, parms, template)


@pytest.mark.parametrize("max_cores", (None, 8))
def test_predict_di_workflow(tmp_path, max_cores):
    """
    Test the Predict DI workflow, using all possible combinations of parameters
    that control the way the CWL workflow is generated from the template.
    Parameters were taken from `PredictDI.set_parset_parameters()`.
    """
    operation = "predict_di"
    template = env_parset.get_template("predict_di_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
    }
    generate_and_validate(tmp_path, operation, parms, template)


@pytest.mark.parametrize("max_cores", (None, 8))
def test_predict_nc_workflow(tmp_path, max_cores):
    """
    Test the Predict NC workflow, using all possible combinations of parameters
    that control the way the CWL workflow is generated from the template.
    Parameters were taken from `PredictNC.set_parset_parameters()`.
    """
    operation = "predict_nc"
    template = env_parset.get_template("predict_nc_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
    }
    generate_and_validate(tmp_path, operation, parms, template)


class TestImageWorkflow:

    operation = "image"
    template = env_parset.get_template("image_pipeline.cwl")
    sub_template = env_parset.get_template("image_sector_pipeline.cwl")

    @pytest.fixture(
        params=generate_keyword_combinations({
            "apply_screens": (False, True),
            "use_facets": (False, True),
            "peel_bright_sources": (False, True),
            "max_cores": (None, 8),
            "use_mpi": (False, True),
            "make_image_cube": (False, True),
            "image_cube_stokes_list": "I",
            "normalize_flux_scale": (False, True),
            "preapply_dde_solutions": (False, True),
            "save_source_list": (False, True),
            "compress_images": (False, True),
        })
    )
    def params(self, request):
        return request.param

    @pytest.fixture()
    def parset(self, params, tmp_path):
        """
        Generate a CWL workflow using `template`, and sub-workflow if
        `sub_template` is given, the same way that
        `rapthor.lib.operation.Operation.setup()` does this.
        """
        pipeline_working_dir = tmp_path / "pipelines" / self.operation
        return create_parsets(pipeline_working_dir, params,
                              self.template, self.sub_template)

    def test_image_workflow(self, params, parset):
        """
        Test the Image workflow, using all possible combinations of parameters
        that control the way the CWL workflow is generated from the template.
        Parameters were taken from `Image.set_parset_parameters()`.
        """
        validate(params, parset)


@pytest.mark.parametrize("max_cores", (None, 8))
@pytest.mark.parametrize("skip_processing", (False, True))
@pytest.mark.parametrize("compress_images", (False, True))
def test_mosaic_workflow(
    tmp_path, max_cores, skip_processing, compress_images
):
    """
    Test the Mosaic workflow, using all possible combinations of parameters
    that control the way the CWL workflow is generated from the template.
    Parameters were taken from `Mosaic.set_parset_parameters()`.
    """
    operation = "mosaic"
    template = env_parset.get_template("mosaic_pipeline.cwl")
    sub_template = env_parset.get_template("mosaic_type_pipeline.cwl")
    parms = {
        "max_cores": max_cores,
        "skip_processing": skip_processing,
        "compress_images": compress_images,
    }
    generate_and_validate(tmp_path, operation, parms, template, sub_template)
