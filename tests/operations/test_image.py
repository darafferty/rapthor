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

    def test_save_model_image(self,field):
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


def parse_cwl_outputs(workflow):
    import yaml
    with open(workflow, 'r') as f:
        return yaml.safe_load(f)['outputs']
    
def generate_mock_files(output_path, outputs, mock_n_files=3):
    for output_info in outputs:
        output_class = output_info["type"]
        for output_source in output_info.get('outputSource', []):
            if output_class == "File[]":
                for idx in range(mock_n_files):
                    output_file = output_path / Path(output_source.replace('/', '.') + f"_{idx}")
                    output_file.touch()
            if output_class == "File":
                output_file = output_path / output_source.replace('/', '.')
                output_file.touch()
            if output_class == "Directory":
                output_dir = output_path / output_source.replace('/', '.')
                output_dir.mkdir(parents=True, exist_ok=True)

from pathlib import Path
def mocked_cwl_execution(self,args, env):
    outputs = parse_cwl_outputs(self.operation.pipeline_parset_file)
    generate_mock_files(Path(self.operation.parset['dir_working']), outputs)
    return True


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

    def test_initial_image_save_model_image(self, field):
        field.parset["imaging_specific"]["save_model_image"] = True
        field.do_predict = False
        field.scan_observations()
        field.define_full_field_sector()
        field.image_pol = 'I'
        image_initial = ImageInitial(field)
        image_initial.set_parset_parameters()
        image_initial.set_input_parameters()

        assert image_initial.input_parms["save_model_image"] is True



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

    def test_save_model_image(self, field):
        field.parset["imaging_specific"]["save_model_image"] = True
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
        assert image_norm.input_parms["save_model_image"] is True

    def test_run_with_execute_mock(self, field, monkeypatch):
        # Mock the CWL runner's execute to avoid spawning subprocesses
        monkeypatch.setattr(
            "rapthor.lib.cwlrunner.BaseCWLRunner.execute",
            mocked_cwl_execution,
        )

        field.parset["imaging_specific"]["save_model_image"] = True
        field.do_predict = False
        field.scan_observations()
        field.define_normalize_sector()
        field.image_pol = 'I'
        field.apply_screens = False
        field.skip_final_major_iteration = False

        image_norm = ImageNormalize(field, index=1)
        image_norm.run()


def test_report_sector_diagnostics(sector_name=None, diagnostics_dict=None, log=None):
    # report_sector_diagnostics(sector_name, diagnostics_dict, log)
    pass
