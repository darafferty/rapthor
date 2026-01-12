"""
Tests for the CWL mock execution utilities.
"""

import tempfile
from pathlib import Path

import pytest

from ..cwl_mock import (
    get_output_type,
    generate_mock_files,
    parse_cwl_outputs,
)


class TestGetOutputType:
    """Tests for get_output_type() function."""

    def test_simple_file_type(self):
        """Test detection of simple File type."""
        output_info = {"type": "File"}
        assert get_output_type(output_info) == "File"

    def test_simple_directory_type(self):
        """Test detection of simple Directory type."""
        output_info = {"type": "Directory"}
        assert get_output_type(output_info) == "Directory"

    def test_array_of_files(self):
        """Test detection of array of files."""
        output_info = {
            "type": {
                "type": "array",
                "items": {"type": "File"}
            }
        }
        assert get_output_type(output_info) == "File[]"

    def test_array_of_directories(self):
        """Test detection of array of directories."""
        output_info = {
            "type": {
                "type": "array",
                "items": {"type": "Directory"}
            }
        }
        assert get_output_type(output_info) == "Directory[]"

    def test_nested_list_files(self):
        """Test detection of nested array of files."""
        output_info = {
            "type": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "File"}
                }
            }
        }
        assert get_output_type(output_info) == "File[][]"

    def test_nested_list_directories(self):
        """Test detection of nested array of directories."""
        output_info = {
            "type": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "Directory"}
                }
            }
        }
        assert get_output_type(output_info) == "Directory[][]"

    def test_array_with_string_items_notation(self):
        """Test array type with simple string notation for items."""
        output_info = {
            "type": {
                "type": "array",
                "items": "File"
            }
        }
        assert get_output_type(output_info) == "File[]"

    def test_nullable_file_type(self):
        """Test nullable File type (File|null)."""
        output_info = {
            "type": ["null", "File"]
        }
        assert get_output_type(output_info) == "File"
        output_info = {
            "type": "File?"
        }
        assert get_output_type(output_info) == "File"
        

    def test_nullable_array_type(self):
        """Test nullable array type (array|null)."""
        output_info = {
            "type": [
                "null",
                {
                    "type": "array",
                    "items": {"type": "File"}
                }
            ]
        }
        assert get_output_type(output_info) == "File[]"


class TestGenerateMockFiles:
    """Tests for generate_mock_files() function."""

    def test_generate_single_file(self):
        """Test generating a single mock file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = [
                {
                    "type": "File",
                    "outputSource": "step/output"
                }
            ]
            generate_mock_files(output_path, outputs)
            
            assert (output_path / "step.output").exists()
            assert (output_path / "step.output").is_file()

    def test_generate_single_directory(self):
        """Test generating a single mock directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = [
                {
                    "type": "Directory",
                    "outputSource": "step/output_dir"
                }
            ]
            generate_mock_files(output_path, outputs)
            
            assert (output_path / "step.output_dir").exists()
            assert (output_path / "step.output_dir").is_dir()

    def test_generate_array_of_files(self):
        """Test generating array of mock files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = [
                {
                    "type": {
                        "type": "array",
                        "items": {"type": "File"}
                    },
                    "outputSource": "step/output_files"
                }
            ]
            generate_mock_files(output_path, outputs, mock_n_outer=3)
            
            # Should create 3 files
            files = sorted(output_path.glob("step.output_files_*"))
            assert len(files) == 3
            assert all(f.is_file() for f in files)

    def test_generate_array_of_directories(self):
        """Test generating array of mock directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = [
                {
                    "type": {
                        "type": "array",
                        "items": {"type": "Directory"}
                    },
                    "outputSource": "step/output_dirs"
                }
            ]
            generate_mock_files(output_path, outputs, mock_n_outer=3)
            
            # Should create 3 directories
            dirs = sorted(output_path.glob("step.output_dirs_*"))
            assert len(dirs) == 3
            assert all(d.is_dir() for d in dirs)

    def test_generate_nested_list_files(self):
        """Test generating nested array of files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = [
                {
                    "type": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "File"}
                        }
                    },
                    "outputSource": "step/nested_files"
                }
            ]
            generate_mock_files(output_path, outputs, mock_n_outer=2, mock_n_inner=2)
            
            # Should create 2 outer directories
            outer_dirs = sorted(output_path.glob("step.nested_files_list_*"))
            assert len(outer_dirs) == 4
            

    def test_generate_nested_list_directories(self):
        """Test generating nested array of directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = [
                {
                    "type": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "Directory"}
                        }
                    },
                    "outputSource": "step/nested_dirs"
                }
            ]
            generate_mock_files(output_path, outputs, mock_n_outer=2, mock_n_inner=2)
            
            # Should create 4 total directories (2 outer times 2 inner)
            outer_dirs = sorted(output_path.glob("step.nested_dirs_list_*"))
            assert len(outer_dirs) == 4
            

    def test_generate_multiple_outputs(self):
        """Test generating multiple different outputs at once."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = [
                {
                    "type": "File",
                    "outputSource": "step1/file_out"
                },
                {
                    "type": "Directory",
                    "outputSource": "step2/dir_out"
                },
                {
                    "type": {
                        "type": "array",
                        "items": {"type": "File"}
                    },
                    "outputSource": "step3/files_out"
                },
            ]
            generate_mock_files(output_path, outputs, mock_n_outer=2, mock_n_inner=2)
            
            assert (output_path / "step1.file_out").is_file()
            assert (output_path / "step2.dir_out").is_dir()
            assert len(list(output_path.glob("step3.files_out_*"))) == 2

    def test_output_source_as_list(self):
        """Test handling outputSource as a list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = [
                {
                    "type": "File",
                    "outputSource": ["step1/out", "step2/out"]
                }
            ]
            generate_mock_files(output_path, outputs)
            
            assert (output_path / "step1.out").is_file()
            assert (output_path / "step2.out").is_file()

    def test_creates_output_path_if_missing(self):
        """Test that output path is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "nested" / "path"
            outputs = [
                {
                    "type": "File",
                    "outputSource": "step/output"
                }
            ]
            generate_mock_files(output_path, outputs)
            
            assert output_path.exists()
            assert (output_path / "step.output").exists()

    def test_custom_mock_n_files(self):
        """Test generating different numbers of mock files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = [
                {
                    "type": {
                        "type": "array",
                        "items": {"type": "File"}
                    },
                    "outputSource": "step/outputs"
                }
            ]
            generate_mock_files(output_path, outputs, mock_n_outer=5)
            
            files = list(output_path.glob("step.outputs_*"))
            assert len(files) == 5


class TestParseOutputsIntegration:
    """Integration tests for parsing real CWL outputs."""

    def test_parse_and_generate_from_cwl_file(self):
        """Test parsing CWL file and generating mocks from it."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a simple CWL file
            cwl_file = Path(tmp_dir) / "workflow.cwl"
            cwl_content = """
cwlVersion: v1.2
class: CommandLineTool
baseCommand: echo
outputs:
  output_file:
    type: File
    outputSource: echo/output
  output_files:
    type:
      type: array
      items: File
    outputSource: echo/outputs
"""
            cwl_file.write_text(cwl_content)
            
            # Parse outputs
            outputs = parse_cwl_outputs(cwl_file)
            
            expected_n_files = 3
            # Generate mocks
            output_path = Path(tmp_dir) / "outputs"
            generate_mock_files(output_path, outputs, mock_n_outer=expected_n_files)
            
            assert (output_path / "echo.output").exists()
            assert len(list(output_path.glob("echo.outputs_*"))) == expected_n_files


class TestMockedCWLExecution:
    """Tests for mocked_cwl_execution() function."""

    def test_mocked_execution_creates_json_output_file(self):
        """Test that mocked_cwl_execution creates JSON outputs file."""
        import json
        from ..cwl_mock import mocked_cwl_execution
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a simple CWL file
            cwl_file = Path(tmp_dir) / "workflow.cwl"
            cwl_content = """
cwlVersion: v1.2
class: CommandLineTool
baseCommand: echo
outputs:
  output_file:
    type: File
    outputSource: echo/output
  output_files:
    type:
      type: array
      items: File
    outputSource: echo/outputs
  output_dir:
    type: Directory
    outputSource: echo/dir
"""
            cwl_file.write_text(cwl_content)
            
            # Create mock parset object
            class MockParset:
                def __getitem__(self, key):
                    if key == 'dir_working':
                        return str(Path(tmp_dir) / "working")
                    raise KeyError(key)
            
            # Create mock operation object
            class MockOperation:
                def __init__(self):
                    self.pipeline_parset_file = str(cwl_file)
                    self.parset = MockParset()
                    self.pipeline_outputs_file = str(Path(tmp_dir) / "working" / "outputs.json")
            
            # Create mock runner object (with operation)
            class MockRunner:
                def __init__(self):
                    self.operation = MockOperation()
            
            # Create working directory
            working_dir = Path(tmp_dir) / "working"
            working_dir.mkdir(parents=True, exist_ok=True)
            
            # Call mocked execution
            runner = MockRunner()
            result = mocked_cwl_execution(runner, [], {})
            
            # Verify result
            assert result is True
            
            # Verify JSON outputs file exists
            outputs_file = Path(runner.operation.pipeline_outputs_file)
            assert outputs_file.exists()
            
            # Verify JSON structure
            with open(outputs_file, 'r') as f:
                outputs = json.load(f)
            
            assert "output" in outputs
            assert "outputs" in outputs
            assert "dir" in outputs
            
            # Verify file output has class and path (matching real CWL format)
            assert isinstance(outputs["output"], dict)
            assert outputs["output"]["class"] == "File"
            assert Path(outputs["output"]["path"]).exists()
            
            # Verify array output is a list of File objects
            assert isinstance(outputs["outputs"], list)
            assert len(outputs["outputs"]) == 3
            for file_obj in outputs["outputs"]:
                assert file_obj["class"] == "File"
                assert "path" in file_obj
            
            # Verify directory output has class and path
            assert isinstance(outputs["dir"], dict)
            assert outputs["dir"]["class"] == "Directory"
            assert Path(outputs["dir"]["path"]).exists()
