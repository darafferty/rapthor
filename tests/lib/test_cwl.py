"""
Test module for testing the CWL workflows _generated_ by the pipeline
"""

import subprocess
import itertools as itt
import os
import tempfile
from pathlib import Path

import pytest
from rapthor.lib.operation import DIR, env_parset
from rapthor.lib.cwl import (
    is_cwl_file,
    is_cwl_directory,
    is_cwl_file_or_directory,
    copy_cwl_object,
    copy_cwl_recursive,
    clean_if_cwl_file_or_directory,
)


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


# Unit tests for CWL utility functions

class TestIsCWLFile:
    """Tests for is_cwl_file() function"""

    def test_valid_file(self):
        """Test with a valid CWL file object"""
        obj = {"class": "File", "path": "/path/to/file.txt"}
        assert is_cwl_file(obj) is True

    def test_valid_directory_not_file(self):
        """Test that CWL directory is not identified as file"""
        obj = {"class": "Directory", "path": "/path/to/dir"}
        assert is_cwl_file(obj) is False

    def test_missing_class_key(self):
        """Test with missing 'class' key"""
        obj = {"path": "/path/to/file.txt"}
        assert is_cwl_file(obj) is False

    def test_wrong_class_value(self):
        """Test with wrong class value"""
        obj = {"class": "InvalidClass", "path": "/path/to/file.txt"}
        assert is_cwl_file(obj) is False

    def test_non_dict_input(self):
        """Test with non-dict input"""
        assert is_cwl_file("not a dict") is False
        assert is_cwl_file([]) is False
        assert is_cwl_file(None) is False


class TestIsCWLDirectory:
    """Tests for is_cwl_directory() function"""

    def test_valid_directory(self):
        """Test with a valid CWL directory object"""
        obj = {"class": "Directory", "path": "/path/to/dir"}
        assert is_cwl_directory(obj) is True

    def test_valid_file_not_directory(self):
        """Test that CWL file is not identified as directory"""
        obj = {"class": "File", "path": "/path/to/file.txt"}
        assert is_cwl_directory(obj) is False

    def test_missing_class_key(self):
        """Test with missing 'class' key"""
        obj = {"path": "/path/to/dir"}
        assert is_cwl_directory(obj) is False

    def test_wrong_class_value(self):
        """Test with wrong class value"""
        obj = {"class": "InvalidClass", "path": "/path/to/dir"}
        assert is_cwl_directory(obj) is False

    def test_non_dict_input(self):
        """Test with non-dict input"""
        assert is_cwl_directory("not a dict") is False
        assert is_cwl_directory([]) is False
        assert is_cwl_directory(None) is False


class TestIsCWLFileOrDirectory:
    """Tests for is_cwl_file_or_directory() function"""

    def test_valid_file(self):
        """Test with a valid CWL file object"""
        obj = {"class": "File", "path": "/path/to/file.txt"}
        assert is_cwl_file_or_directory(obj) is True

    def test_valid_directory(self):
        """Test with a valid CWL directory object"""
        obj = {"class": "Directory", "path": "/path/to/dir"}
        assert is_cwl_file_or_directory(obj) is True

    def test_invalid_class(self):
        """Test with invalid class"""
        obj = {"class": "InvalidClass", "path": "/path/to/something"}
        assert is_cwl_file_or_directory(obj) is False

    def test_non_dict_input(self):
        """Test with non-dict input"""
        assert is_cwl_file_or_directory("not a dict") is False
        assert is_cwl_file_or_directory([]) is False
        assert is_cwl_file_or_directory(None) is False


class TestCopyCWLObject:
    """Tests for copy_cwl_object() function"""

    def test_copy_file(self, tmp_path):
        """Test copying a CWL file object"""
        # Create a source file
        src_file = tmp_path / "source" / "test.txt"
        src_file.parent.mkdir(parents=True, exist_ok=True)
        src_file.write_text("test content")

        # Create CWL file object
        cwl_obj = {"class": "File", "path": str(src_file)}

        # Copy to destination
        dest_dir = tmp_path / "destination"
        dest_dir.mkdir(parents=True, exist_ok=True)
        copy_cwl_object(cwl_obj, str(dest_dir))

        # Verify file was copied
        dest_file = dest_dir / "test.txt"
        assert dest_file.exists()
        assert dest_file.read_text() == "test content"

    def test_copy_directory(self, tmp_path):
        """Test copying a CWL directory object"""
        # Create a source directory with files
        src_dir = tmp_path / "source_dir"
        src_dir.mkdir()
        (src_dir / "file1.txt").write_text("content1")
        (src_dir / "file2.txt").write_text("content2")

        # Create CWL directory object
        cwl_obj = {"class": "Directory", "path": str(src_dir)}

        # Copy to destination
        dest_parent = tmp_path / "destination"
        copy_cwl_object(cwl_obj, str(dest_parent))

        # Verify directory was copied
        dest_dir = dest_parent / "source_dir"
        assert dest_dir.exists()
        assert (dest_dir / "file1.txt").read_text() == "content1"
        assert (dest_dir / "file2.txt").read_text() == "content2"

    def test_copy_non_cwl_object(self, tmp_path):
        """Test that non-CWL objects are silently ignored"""
        # Should not raise an exception
        copy_cwl_object({"not": "cwl"}, str(tmp_path))
        copy_cwl_object("string", str(tmp_path))
        copy_cwl_object(None, str(tmp_path))


class TestCopyCWLRecursive:
    """Tests for copy_cwl_recursive() function"""

    def test_copy_single_file(self, tmp_path):
        """Test recursive copy of a single CWL file"""
        src_file = tmp_path / "source" / "test.txt"
        src_file.parent.mkdir(parents=True, exist_ok=True)
        src_file.write_text("test")

        cwl_obj = {"class": "File", "path": str(src_file)}
        dest_dir = tmp_path / "destination"
        dest_dir.mkdir(parents=True, exist_ok=True)

        copy_cwl_recursive(cwl_obj, str(dest_dir))

        assert (dest_dir / "test.txt").exists()
        assert (dest_dir / "test.txt").read_text() == "test"

    def test_copy_list_of_files(self, tmp_path):
        """Test recursive copy of a list of CWL files"""
        # Create source files
        src1 = tmp_path / "source" / "file1.txt"
        src2 = tmp_path / "source" / "file2.txt"
        src1.parent.mkdir(parents=True, exist_ok=True)
        src1.write_text("content1")
        src2.write_text("content2")

        cwl_list = [
            {"class": "File", "path": str(src1)},
            {"class": "File", "path": str(src2)},
        ]
        dest_dir = tmp_path / "destination"
        dest_dir.mkdir(parents=True, exist_ok=True)

        copy_cwl_recursive(cwl_list, str(dest_dir))

        assert (dest_dir / "file1.txt").exists()
        assert (dest_dir / "file2.txt").exists()
        assert (dest_dir / "file1.txt").read_text() == "content1"
        assert (dest_dir / "file2.txt").read_text() == "content2"

    def test_copy_nested_list(self, tmp_path):
        """Test recursive copy of nested lists"""
        src1 = tmp_path / "source" / "file1.txt"
        src2 = tmp_path / "source" / "file2.txt"
        src1.parent.mkdir(parents=True, exist_ok=True)
        src1.write_text("content1")
        src2.write_text("content2")

        nested_list = [
            [{"class": "File", "path": str(src1)}],
            [{"class": "File", "path": str(src2)}],
        ]
        dest_dir = tmp_path / "destination"
        dest_dir.mkdir(parents=True, exist_ok=True)

        copy_cwl_recursive(nested_list, str(dest_dir))

        assert (dest_dir / "file1.txt").exists()
        assert (dest_dir / "file2.txt").exists()
        assert (dest_dir / "file1.txt").read_text() == "content1"
        assert (dest_dir / "file2.txt").read_text() == "content2"

    def test_copy_non_cwl_in_list(self, tmp_path):
        """Test that non-CWL objects in lists are ignored"""
        src_file = tmp_path / "source" / "test.txt"
        src_file.parent.mkdir(parents=True, exist_ok=True)
        src_file.write_text("test")

        mixed_list = [
            {"class": "File", "path": str(src_file)},
            "not a cwl object",
            {"not": "cwl"},
        ]
        dest_dir = tmp_path / "destination"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Should not raise an exception
        copy_cwl_recursive(mixed_list, str(dest_dir))

        # CWL file should still be copied
        assert (dest_dir / "test.txt").exists()
        assert (dest_dir / "test.txt").read_text() == "test"


class TestCleanIfCWLFileOrDirectory:
    """Tests for clean_if_cwl_file_or_directory() function"""

    def test_remove_file(self, tmp_path):
        """Test removing a CWL file object"""
        src_file = tmp_path / "test.txt"
        src_file.write_text("test")

        cwl_obj = {"class": "File", "path": str(src_file)}

        assert src_file.exists()
        clean_if_cwl_file_or_directory(cwl_obj)
        assert not src_file.exists()

    def test_remove_directory(self, tmp_path):
        """Test removing a CWL directory object"""
        src_dir = tmp_path / "test_dir"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("content")

        cwl_obj = {"class": "Directory", "path": str(src_dir)}

        assert src_dir.exists()
        clean_if_cwl_file_or_directory(cwl_obj)
        assert not src_dir.exists()

    def test_remove_list_of_files(self, tmp_path):
        """Test removing a list of CWL file objects"""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        cwl_list = [
            {"class": "File", "path": str(file1)},
            {"class": "File", "path": str(file2)},
        ]

        assert file1.exists() and file2.exists()
        clean_if_cwl_file_or_directory(cwl_list)
        assert not file1.exists() and not file2.exists()

    def test_remove_nonexistent_file(self):
        """Test that removing nonexistent files doesn't raise exception"""
        cwl_obj = {"class": "File", "path": "/nonexistent/path/file.txt"}

        # Should not raise an exception
        clean_if_cwl_file_or_directory(cwl_obj)

    def test_remove_non_cwl_object(self, tmp_path):
        """Test that non-CWL objects are silently ignored"""
        # Should not raise exceptions
        clean_if_cwl_file_or_directory({"not": "cwl"})
        clean_if_cwl_file_or_directory("string")
        clean_if_cwl_file_or_directory(None)

