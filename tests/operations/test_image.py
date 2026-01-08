"""
Test cases for the `rapthor.operations.image` module.
"""

import pytest
from rapthor.operations.image import (Image, ImageInitial, ImageNormalize,
                                      report_sector_diagnostics)

from rapthor.lib.strategy import set_selfcal_strategy
from rapthor.process import chunk_observations

@pytest.fixture
def image(field, index=1):
    """
    Create an instance of the Image operation.
    """
    # return Image(field, index=index)
    return "mock_image"


@pytest.fixture
def image_initial(field, index=1):
    """
    Create an instance of the ImageInitial operation.
    """
    # return ImageInitial(field, index=index)
    return "mock_image_initial"


@pytest.fixture
def image_normalize(field, index=1):
    """
    Create an instance of the ImageNormalize operation.
    """
    # return ImageNormalize(field, index=index)
    return "mock_image_normalize"


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

def test_save_model_image(field):
    # This is the required setup to configure an Image operation
    # avoiding any other setting will make it throw an expeception
    # refactoring of the fild and image classes seems advisable here
    field.parset["imaging_specific"]["save_model_image"] = True
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
    
    assert image.input_parms["save_model_image"] is True

    pass

class TestImageInitial:
    def test_set_parset_parameters(self, image_initial):
        # image_initial.set_parset_parameters()
        pass

    def test_set_input_parameters(self, image_initial):
        # image_initial.set_input_parameters()
        pass

    def test_finalize(self, image_initial):
        # image_initial.finalize()
        pass


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


def test_report_sector_diagnostics(sector_name=None, diagnostics_dict=None, log=None):
    # report_sector_diagnostics(sector_name, diagnostics_dict, log)
    pass
