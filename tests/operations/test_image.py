"""
Test cases for the `rapthor.operations.image` module.
"""

import pytest

from pathlib import Path
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
def expected_image_output_last_cycle():
    """
    Fixture to provide expected output structure for CWL execution in the last cycle.
    """
    return {
        "sector_I_images": [["sector0-MFS-I-image-pb.fits", "sector0-MFS-I-image.fits"]],
        "filtered_skymodel_true_sky": ["sector0.true_sky.txt"],
        "filtered_skymodel_apparent_sky": ["sector0.apparent_sky.txt"],
        "pybdsf_catalog": ["sector0.source_catalog.fits"],
        "sector_diagnostics": ["sector0_diagnostics.json"],
        "sector_offsets": ["sector0_offsets.txt"],
        "source_filtering_mask": ["sector0_mask.fits"],
        "sector_extra_images": [[
            'sector_1-MFS-I-image-pb.fits',
            'sector_1-MFS-I-image-pb.fits',
            'sector_1-MFS-I-image.fits',
            'sector_1-MFS-Q-image.fits',
            'sector_1-MFS-U-image.fits',
            'sector_1-MFS-V-image.fits',
            'sector_1-MFS-Q-image-pb.fits',
            'sector_1-MFS-U-image-pb.fits',
            'sector_1-MFS-V-image-pb.fits',
            'sector_1-MFS-I-residual.fits',
            'sector_1-MFS-Q-residual.fits',
            'sector_1-MFS-U-residual.fits',
            'sector_1-MFS-V-residual.fits',
            'sector_1-MFS-I-model-pb.fits',
            'sector_1-MFS-Q-model-pb.fits',
            'sector_1-MFS-U-model-pb.fits',
            'sector_1-MFS-V-model-pb.fits',
            'sector_1-MFS-I-dirty.fits',
            'sector_1-MFS-Q-dirty.fits',
            'sector_1-MFS-U-dirty.fits',
            'sector_1-MFS-V-dirty.fits'
        ]]
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
def image_last_cycle(field, monkeypatch, expected_image_output_last_cycle):
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
        lambda self, args, env: mocked_cwl_execution(self, args, env, expected_image_output_last_cycle),
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
        assert image.is_done()
        image.outputs["sector_diagnostic_plots"][0] = None  # Simulate missing diagnostic plots
        # Handles missing diagnostic plots gracefully without raising an exception
        image.finalize()

    @pytest.mark.parametrize("save_visibilities", [True, False])
    def test_finalize_save_visibilities(self, image, save_visibilities):
        image.field.save_visibilities = save_visibilities
        image.run()
        assert image.is_done()

    def test_sector_extra_images_on_last_cycle(self, image_last_cycle):
        image_last_cycle.run()
        assert image_last_cycle.is_done()
        # Check that the expected I, Q, U, V images are in the outputs
        sector_0 = image_last_cycle.field.imaging_sectors[0]
        assert sector_0.name == "sector_1", f"Expected sector name 'sector_1', got '{sector_0.name}'"
        for pol in ['I', 'Q', 'U', 'V']:
            assert hasattr(
                sector_0, f"{pol}_image_file_true_sky"), f"Expected {pol}_image_file_true_sky to be set in sector_1"
     
    def test_sector_save_supplementary_images_null_mask(self, image):
        image.field.save_supplementary_images = True
        image.set_input_parameters()
        image.run()
        assert image.is_done()
        # Simulate a null mask output and check that it is handled gracefully
        image.outputs["source_filtering_mask"] = [None] 
        sector_0 = image.field.imaging_sectors[0]
        assert hasattr(sector_0, "mask_filename"), "Expected mask_filename to be set in sector_1"
    
    def test_sector_save_supplementary_images(self, image):
        image.field.save_supplementary_images = True
        image.set_input_parameters()
        image.run()
        assert image.is_done()
        sector_0 = image.field.imaging_sectors[0]
        assert hasattr(sector_0, "mask_filename"), "Expected mask_filename to be set in sector_1"
        assert isinstance(sector_0.mask_filename, (str, Path)), f"Expected mask_filename to be a string, got {type(sector_0.mask_filename)}"
    def test_find_in_file_list(self):
        # Test the find_in_file_list method with a sample file list
        file_list = [
            'sector_1-MFS-I-image-pb.fits',
            'sector_1-MFS-I-image.fits',
            'sector_1-MFS-Q-image-pb.fits',
            'sector_1-MFS-Q-image.fits',
            'sector_1-MFS-U-image-pb.fits',
            'sector_1-MFS-U-image.fits',
            'sector_1-MFS-V-image-pb.fits',
            'sector_1-MFS-V-image.fits'
        ]
        type_path_map = Image.find_in_file_list(file_list)
        expected_map = {
            "image_file_true_sky": ['sector_1-MFS-I-image-pb.fits', 'sector_1-MFS-Q-image-pb.fits', 'sector_1-MFS-U-image-pb.fits', 'sector_1-MFS-V-image-pb.fits'],
            "image_file_apparent_sky": ['sector_1-MFS-I-image.fits', 'sector_1-MFS-Q-image.fits', 'sector_1-MFS-U-image.fits', 'sector_1-MFS-V-image.fits']
        }
        assert type_path_map == expected_map, f"Expected {expected_map}, got {type_path_map}"

    @pytest.mark.parametrize("pol", ["I", "Q", "U", "V", "X"])
    def test_derive_pol_from_filename(self, pol):
        filename = f'sector_1-MFS-{pol}-image-pb.fits'
        derived_pol = Image.derive_pol_from_filename(filename)
        expected_pol = pol if pol in "IQUV" else "I"
        assert derived_pol == expected_pol, f"Expected polarization '{expected_pol}', got '{derived_pol}'"


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

    @pytest.mark.parametrize("dde_method", ["single", "full"])
    def test_initial_image_with_dde_method_single_does_not_raise(self, field, dde_method):
        """
        Regression test for AttributeError when generate_initial_image=True.
        """
        # Set dde_method to 'single' to trigger the bug condition
        field.parset["imaging_specific"]["dde_method"] = dde_method
        field.dde_method = dde_method  # Ensure the field attribute is also set
        field.do_predict = False
        field.scan_observations()
        field.define_full_field_sector()
        field.image_pol = 'I'
        
        image_initial = ImageInitial(field)
        
        # This should NOT raise AttributeError: 'Sector' object has no attribute 'central_patch'
        # The bug causes this to fail because preapply_dde_solutions is incorrectly True
        image_initial.set_parset_parameters()
        image_initial.set_input_parameters()
        
        # Verify apply_none is True and preapply_dde_solutions is False
        assert image_initial.apply_none is True, "apply_none should be True for ImageInitial"
        assert image_initial.preapply_dde_solutions is False, \
            "preapply_dde_solutions should be False for ImageInitial even with dde_method='single'"

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
        assert image_initial.is_done()
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
