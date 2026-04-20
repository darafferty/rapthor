import pytest

from tests.cwl.cwl_mock import mocked_cwl_execution
from rapthor.lib.strategy import set_selfcal_strategy

from rapthor.operations.image import Image
from rapthor.operations.mosaic import Mosaic

"""
Integration tests from the Image operation to the Mosaic operation.

Currently the execution in process of a step has the following 3 stages:
1. set_input_parameters: where the input parameters are set based on the
   field and the index
2. set_parset_parameters: where the parset parameters are set based on
   the field and the index
3. run: where the actual execution of the operation happens, which
   includes the execution of the CWL workflow.
During the run stage, the CWL execution is mocked to return the expected
output for the Image operation. Upon successful completion, the Image
operation sets various attributes in the field using the mocked CWL output.
The Mosaic operation uses the updated field to set its input parameters
and parset parameters.
"""

@pytest.fixture
def field_I_no_predict(field):
    """
    Fixture to set up the field for the image operation with predict disabled.
    This fixture modifies the field to disable the predict step and sets the
    image polarization to 'I'.
    With the testing skymodel this is the only way to avoid getting an error in the clustering
    of the sources performed in the Field.update() method.
    """
    field.h5parm_filename = "nonexisting_h5parm_file.h5"
    field.scan_observations()
    field.parset["regroup_input_skymodel"] = False
    steps = set_selfcal_strategy(field)
    field.update(steps[0], index=1, final=False)
    # The field update will set the predict flag to True, override it here
    field.do_predict = False
    field.image_pol = 'I'
    field.skip_final_major_iteration = True
    return field


@pytest.mark.integration
def test_image_I_to_mosaic(field_I_no_predict, expected_image_output, monkeypatch):
    """
    Test the execution of the following operations in sequence:
    1. Image (with predict disabled)
    2. Mosaic (with the output of the Image operation as input)
    """
    # Patch the CWL execution to return the expected output for the Image operation
    monkeypatch.setattr(
        "rapthor.lib.cwlrunner.BaseCWLRunner.execute",
        lambda self, args, env: mocked_cwl_execution(self, args, env, expected_image_output),
        raising=False
    )
    image = Image(field=field_I_no_predict, index=1)
    image.set_input_parameters()
    image.set_parset_parameters()
    image.run()
    # Now create and run the Mosaic operation (after sector image attributes are set)
    monkeypatch.setattr(
        "rapthor.lib.cwlrunner.BaseCWLRunner.execute",
        lambda self, args, env: mocked_cwl_execution(self, args, env),
        raising=False
    )

    mosaic = Mosaic(field=image.field, index=1)
    mosaic.set_input_parameters()
    mosaic.set_parset_parameters()
    mosaic.run()
