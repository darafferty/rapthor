import pytest

from cwl.cwl_mock import mocked_cwl_execution
from rapthor.lib.strategy import set_selfcal_strategy

from rapthor.operations.image import Image
from rapthor.operations.mosaic import Mosaic

@pytest.fixture
def field_I_no_predict(field):
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

@pytest.fixture
def image_patched_execution(field_I_no_predict, monkeypatch, expected_image_output):
    """                       
    Fixture to patch the CWL execution for the Image operation.
    """                             
    monkeypatch.setattr(
            "rapthor.lib.cwlrunner.BaseCWLRunner.execute",
            lambda self, args, env: mocked_cwl_execution(self, args, env, expected_image_output),
            raising=False
        )
    image = Image(field=field_I_no_predict, index=1)
    return image

@pytest.mark.integration
def test_image_to_mosaic(image_patched_execution, monkeypatch):
    """
    Test the `image_to_mosaic` operation.
    """
    # Run the Image operation first to set sector attributes
    image_patched_execution.run()

    # Now create and run the Mosaic operation (after sector image attributes are set)
    with monkeypatch.context() as m:
        m.setattr(
            "rapthor.lib.cwlrunner.BaseCWLRunner.execute",
            lambda self, args, env: mocked_cwl_execution(self, args, env),
            raising=False
        )
        image_patched_execution.set_input_parameters()
        image_patched_execution.set_parset_parameters()
        image_patched_execution.run()

        mosaic = Mosaic(field=image_patched_execution.field, index=1)
        mosaic.set_input_parameters()
        mosaic.set_parset_parameters()
        mosaic.run()