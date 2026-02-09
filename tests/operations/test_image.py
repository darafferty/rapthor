"""
Test cases for the `rapthor.operations.image` module.
"""

import pytest

from cwl.cwl_mock import mocked_cwl_execution
from rapthor.lib.strategy import set_selfcal_strategy

from rapthor.operations.image import Image, ImageInitial, ImageNormalize


@pytest.fixture
def expected_image_output():
    """
    Fixture to provide expected output structure for CWL execution.
    """
    return {
        "sector_I_images": [["sector0-MFS-I-image-pb.fits", "sector0-MFS-I-image.fits"]],
        "sector_extra_images": [["sector0-MFS-I-residual.fits", "sector0-MFS-I-model-pb.fits"]],
        "filtered_skymodel_true_sky": ["sector0.true_sky.txt"],
        "filtered_skymodel_apparent_sky": ["sector0.apparent_sky.txt"],
        "pybdsf_catalog": ["sector0.source_catalog.fits"],
        "sector_diagnostics": ["sector0_diagnostics.json"],
        "sector_offsets": ["sector0_offsets.txt"],
    }


@pytest.fixture
def image(field, monkeypatch, expected_image_output):
    """
    Fixture to mock CWL execution for the Image operation.
    Create an instance of the Image operation.
    """
    # Set the required attributes directly without running strategy setup
    # which would do real processing
    field.parset["regroup_input_skymodel"] = False
    # Since we are not doing the calibration provide a non-existing h5parm file
    field.h5parm_filename = "nonexisting_h5parm_file.h5"
    field.scan_observations()
    steps = set_selfcal_strategy(field)
    field.update(steps[0], index=1, final=False)
    # The field update will set the predict flag to True, override it here
    field.do_predict = False
    field.image_pol = 'I'
    field.skip_final_major_iteration = True

    # Mock the execute method on the instance
    monkeypatch.setattr(
        "rapthor.lib.cwlrunner.BaseCWLRunner.execute",
        lambda self, args, env: mocked_cwl_execution(self, args, env, expected_image_output),
        raising=False
    )
    image = Image(field=field, index=1)

    image.set_parset_parameters()
    image.set_input_parameters()

    return image


@pytest.fixture
def image_initial(field, monkeypatch, expected_image_output):
    """
    Create an instance of the ImageInitial operation.
    """
    # Mock the execute method on the instance
    monkeypatch.setattr(
        "rapthor.lib.cwlrunner.BaseCWLRunner.execute",
        lambda self, args, env: mocked_cwl_execution(self, args, env, expected_outputs=expected_image_output),
        raising=False
    )
    field.do_predict = False
    field.scan_observations()
    field.define_full_field_sector()
    field.image_pol = 'I'
    image_initial = ImageInitial(field)
    image_initial.set_parset_parameters()
    image_initial.set_input_parameters()

    return image_initial


@pytest.fixture
def image_normalize(field, monkeypatch, expected_image_output):
    """
    Create an instance of the ImageNormalize operation.
    """
    # Mock the execute method on the instance
    monkeypatch.setattr(
        "rapthor.lib.cwlrunner.BaseCWLRunner.execute",
        lambda self, args, env: mocked_cwl_execution(self, args, env, expected_outputs=expected_image_output),
        raising=False
    )
    field.do_predict = False
    field.scan_observations()
    field.define_normalize_sector()
    field.image_pol = 'I'
    field.apply_screens = False
    field.skip_final_major_iteration = False
    image_norm = ImageNormalize(field, index=1)
    image_norm.do_predict = False
    image_norm.set_parset_parameters()
    image_norm.set_input_parameters()
    return image_norm


class TestImage:
    def test_set_parset_parameters(self, image):
        # image.set_parset_parameters()
        pass

    def test_set_input_parameters(self, image):
        # image.set_input_parameters()
        pass

    def test_finalize(self, image):
        # image.finalize()
        pass

    def test_run(self, image):
        image.run()
        assert image.is_done()

    def test_save_model_image(self, field):
        # This is the required setup to configure an Image operation
        # avoiding any other setting will make it throw an expeception
        # refactoring of the fild and image classes seems advisable here
        field.parset["imaging_specific"]["save_filtered_model_image"] = True
        field.parset["regroup_input_skymodel"] = False
        field.do_predict = False
        field.scan_observations()
        steps = set_selfcal_strategy(field)
        field.update(steps[0], index=1, final=False)
        field.image_pol = 'I'
        field.skip_final_major_iteration = True
        image = Image(field, index=1)
        image.do_predict = False
        image.apply_none = True
        image.set_parset_parameters()
        image.set_input_parameters()

        assert image.input_parms["save_filtered_model_image"] is True

    def test_finalize_without_diagnostic_plots(self, image):
        image.run()
        image.is_done()
        image.outputs["sector_diagnostic_plots"][0] = None  # Simulate missing diagnostic plots
        # Handles missing diagnostic plots gracefully without raising an exception
        image.finalize()

    @pytest.mark.parametrize("save_visibilities", [True, False])
    def test_finalize_save_visibilities(self, image, save_visibilities):
        image.field.save_visibilities = save_visibilities
        image.run()
        image.finalize()


class TestImageInitial:
    def test_set_parset_parameters(self, image_initial):
        # image_initial.set_parset_parameters()
        pass

    def test_set_input_parameters(self, image_initial):
        # image_initial.set_input_parameters()
        pass

    def test_run(self, image_initial):
        image_initial.run()
        assert image_initial.is_done()

    def test_initial_image_save_model_image(self, field):
        field.parset["imaging_specific"]["save_filtered_model_image"] = True
        field.do_predict = False
        field.scan_observations()
        field.define_full_field_sector()
        field.image_pol = 'I'
        image_initial = ImageInitial(field)
        image_initial.set_parset_parameters()
        image_initial.set_input_parameters()

        assert image_initial.input_parms["save_filtered_model_image"] is True

    def test_finalize_without_diagnostic_plots(self, image_initial):
        image_initial.run()
        image_initial.is_done()
        # Simulate missing diagnostic plots
        image_initial.outputs["sector_diagnostic_plots"][0] = None
        # Handles missing diagnostic plots gracefully without raising
        # an exception
        image_initial.finalize()


class TestImageNormalize:
    def test_set_parset_parameters(self, image_normalize):
        # image_normalize.set_parset_parameters()
        pass

    def test_set_input_parameters(self, image_normalize):
        # image_normalize.set_input_parameters()
        pass

    def test_finalize(self, image_normalize):
        # image_normalize.finalize()
        pass

    def test_run(self, image_normalize):
        image_normalize.run()
        assert image_normalize.is_done()

    def test_save_model_image(self, field):
        field.parset["imaging_specific"]["save_filtered_model_image"] = True
        field.do_predict = False
        field.scan_observations()
        field.define_normalize_sector()
        field.image_pol = 'I'
        field.apply_screens = False
        field.skip_final_major_iteration = False
        image_norm = ImageNormalize(field, index=1)
        image_norm.do_predict = False
        image_norm.set_parset_parameters()
        image_norm.set_input_parameters()
        assert image_norm.input_parms["save_filtered_model_image"] is True

    def test_run_with_execute_mock(self, field):
        field.parset["imaging_specific"]["save_filtered_model_image"] = True
        field.do_predict = False
        field.scan_observations()
        field.define_normalize_sector()
        field.image_pol = 'I'
        field.apply_screens = False
        field.skip_final_major_iteration = False


def test_report_sector_diagnostics(sector_name=None, diagnostics_dict=None, log=None):
    # report_sector_diagnostics(sector_name, diagnostics_dict, log)
    pass
