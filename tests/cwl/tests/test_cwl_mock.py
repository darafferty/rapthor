"""
Tests for the CWL mock execution utilities.
"""

import tempfile
from pathlib import Path
import yaml
from textwrap import dedent
from ..cwl_mock import (
    load_cwl_workflow,
    infer_output_type,
    build_mock_outputs,
    extract_cwl_outputs,
    mocked_cwl_execution
)


class TestGetOutputType:
    """Tests for infer_output_type() function."""

    def test_simple_file_type(self):
        """Test detection of simple File type."""
        output_info = {"type": "File"}
        assert infer_output_type(output_info) == "File"

    def test_simple_directory_type(self):
        """Test detection of simple Directory type."""
        output_info = {"type": "Directory"}
        assert infer_output_type(output_info) == "Directory"

    def test_array_of_files(self):
        """Test detection of array of files."""
        output_info = {
            "type": {
                "type": "array",
                "items": {"type": "File"}
            }
        }
        assert infer_output_type(output_info) == "File[]"

    def test_array_of_directories(self):
        """Test detection of array of directories."""
        output_info = {
            "type": {
                "type": "array",
                "items": {"type": "Directory"}
            }
        }
        assert infer_output_type(output_info) == "Directory[]"

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
        assert infer_output_type(output_info) == "File[][]"

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
        assert infer_output_type(output_info) == "Directory[][]"

    def test_array_with_string_items_notation(self):
        """Test array type with simple string notation for items."""
        output_info = {
            "type": {
                "type": "array",
                "items": "File"
            }
        }
        assert infer_output_type(output_info) == "File[]"

    def test_nullable_file_type(self):
        """Test nullable File type (File|null)."""
        output_info = {
            "type": ["null", "File"]
        }
        assert infer_output_type(output_info) == "File"
        output_info = {
            "type": "File?"
        }
        assert infer_output_type(output_info) == "File"
        

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
        assert infer_output_type(output_info) == "File[]"


class TestGenerateMockFiles:
    """Tests for build_mock_outputs() function."""

    def test_generate_single_file(self):
        """Test generating a single mock file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = {
                "output": {
                    "type": "File",
                    "outputSource": "step/output"
                }
            }
            result = build_mock_outputs(output_path, outputs)
            
            assert "output" in result
            assert isinstance(result["output"], dict)
            assert result["output"]["class"] == "File"
            assert "path" in result["output"]

    def test_generate_single_directory(self):
        """Test generating a single mock directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = {
                "output_dir": {
                    "type": "Directory",
                    "outputSource": "step/output_dir"
                }
            }
            result = build_mock_outputs(output_path, outputs)
            
            assert "output_dir" in result
            assert isinstance(result["output_dir"], dict)
            assert result["output_dir"]["class"] == "Directory"
            assert "path" in result["output_dir"]

    def test_generate_array_of_files(self):
        """Test generating array of mock files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = {
                "output_files": {
                    "type": {
                        "type": "array",
                        "items": {"type": "File"}
                    },
                    "outputSource": "step/output_files"
                }
            }
            result = build_mock_outputs(output_path, outputs, mock_n_outer=3)
            
            assert "output_files" in result
            assert isinstance(result["output_files"], list)
            assert len(result["output_files"]) == 3
            for file_obj in result["output_files"]:
                assert isinstance(file_obj, dict)
                assert file_obj["class"] == "File"
                assert "path" in file_obj

    def test_generate_array_of_directories(self):
        """Test generating array of mock directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = {
                "output_dirs": {
                    "type": {
                        "type": "array",
                        "items": {"type": "Directory"}
                    },
                    "outputSource": "step/output_dirs"
                }
            }
            result = build_mock_outputs(output_path, outputs, mock_n_outer=3)
            
            assert "output_dirs" in result
            assert isinstance(result["output_dirs"], list)
            assert len(result["output_dirs"]) == 3
            for dir_obj in result["output_dirs"]:
                assert isinstance(dir_obj, dict)
                assert dir_obj["class"] == "Directory"
                assert "path" in dir_obj

    def test_generate_nested_list_files(self):
        """Test generating nested array of files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = {
                "nested_files": {
                    "type": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "File"}
                        }
                    },
                    "outputSource": "step/nested_files"
                }
            }
            result = build_mock_outputs(output_path, outputs, mock_n_outer=2, mock_n_inner=2)
            
            assert "nested_files" in result
            assert isinstance(result["nested_files"], list)
            assert len(result["nested_files"]) == 2
            for outer_list in result["nested_files"]:
                assert isinstance(outer_list, list)
                assert len(outer_list) == 2
                for file_obj in outer_list:
                    assert isinstance(file_obj, dict)
                    assert file_obj["class"] == "File"
                    assert "path" in file_obj
            

    def test_generate_nested_list_directories(self):
        """Test generating nested array of directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = {
                "nested_dirs": {
                    "type": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "Directory"}
                        }
                    },
                    "outputSource": "step/nested_dirs"
                }
            }
            result = build_mock_outputs(output_path, outputs, mock_n_outer=2, mock_n_inner=2)
            
            assert "nested_dirs" in result
            assert isinstance(result["nested_dirs"], list)
            assert len(result["nested_dirs"]) == 2
            for outer_list in result["nested_dirs"]:
                assert isinstance(outer_list, list)
                assert len(outer_list) == 2
                for dir_obj in outer_list:
                    assert isinstance(dir_obj, dict)
                    assert dir_obj["class"] == "Directory"
                    assert "path" in dir_obj
            

    def test_generate_multiple_outputs(self):
        """Test generating multiple different outputs at once."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = {
                "file_out": {
                    "type": "File",
                    "outputSource": "step1/file_out"
                },
                "dir_out": {
                    "type": "Directory",
                    "outputSource": "step2/dir_out"
                },
                "files_out": {
                    "type": {
                        "type": "array",
                        "items": {"type": "File"}
                    },
                    "outputSource": "step3/files_out"
                },
            }
            result = build_mock_outputs(output_path, outputs, mock_n_outer=2, mock_n_inner=2)
            
            assert "file_out" in result
            assert result["file_out"]["class"] == "File"
            assert "dir_out" in result
            assert result["dir_out"]["class"] == "Directory"
            assert "files_out" in result
            assert isinstance(result["files_out"], list)
            assert len(result["files_out"]) == 2

    def test_output_source_as_list(self):
        """Test handling outputSource as a list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = {
                "out": {
                    "type": "File",
                    "outputSource": ["step1/out", "step2/out"]
                }
            }
            result = build_mock_outputs(output_path, outputs)
            
            assert "out" in result
            assert isinstance(result["out"], list)
            assert len(result["out"]) == 2
            for file_obj in result["out"]:
                assert file_obj["class"] == "File"
                assert "path" in file_obj

    def test_creates_output_path_if_missing(self):
        """Test that output path is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "nested" / "path"
            outputs = {
                "output": {
                    "type": "File",
                    "outputSource": "step/output"
                }
            }
            result = build_mock_outputs(output_path, outputs)
            
            assert output_path.exists()
            assert "output" in result
            assert result["output"]["class"] == "File"

    def test_custom_mock_n_files(self):
        """Test generating different numbers of mock files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            outputs = {
                "outputs": {
                    "type": {
                        "type": "array",
                        "items": {"type": "File"}
                    },
                    "outputSource": "step/outputs"
                }
            }
            result = build_mock_outputs(output_path, outputs, mock_n_outer=5)
            
            assert "outputs" in result
            assert isinstance(result["outputs"], list)
            assert len(result["outputs"]) == 5


class TestParseOutputsIntegration:
    """Integration tests for parsing real CWL outputs."""

    def test_parse_and_generate_from_cwl_file(self):
        """Test parsing CWL file and generating mocks from it."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a simple CWL file
            cwl_file = Path(tmp_dir) / "workflow.cwl"
            cwl_content = dedent("""
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
            """)
            cwl_file.write_text(cwl_content)
            
            # Parse outputs
            outputs = extract_cwl_outputs(cwl_file)
            
            expected_n_files = 3
            # Generate mocks
            output_path = Path(tmp_dir) / "outputs"
            result = build_mock_outputs(output_path, outputs, mock_n_outer=expected_n_files)
            
            # Check files exist on disk
            assert (output_path / "echo.output").exists()
            assert len(list(output_path.glob("echo.outputs_*"))) == expected_n_files
            
            # Check returned JSON structure
            assert "output_file" in result
            assert isinstance(result["output_file"], dict)
            assert result["output_file"]["class"] == "File"
            assert "path" in result["output_file"]
            
            # Verify the JSON path points to the actual file on disk
            output_file_path = Path(result["output_file"]["path"])
            assert output_file_path.exists()
            assert output_file_path.is_file()
            
            # Check array output structure
            assert "output_files" in result
            assert isinstance(result["output_files"], list)
            assert len(result["output_files"]) == expected_n_files
            
            # Verify each file in the array exists on disk
            for i, file_obj in enumerate(result["output_files"]):
                assert isinstance(file_obj, dict)
                assert file_obj["class"] == "File"
                assert "path" in file_obj
                file_path = Path(file_obj["path"])
                assert file_path.exists(), f"File {file_obj['path']} does not exist"
                assert file_path.is_file()

    def test_load_cwl_workflow(self, tmp_path):
        """Test that load_cwl_workflow handles both list and dict step formats correctly.
        
        When steps are defined as a dict, they should be converted to a list with 'id' keys.
        When steps are defined as a list, they should be returned as-is.
        Both formats should produce equivalent results.
        """
        # Create a workflow with steps as a list
        workflow_list = {
            'cwlVersion': 'v1.0',
            'class': 'Workflow',
            'inputs': {
                'input1': 'string'
            },
            'outputs': {
                'output1': {
                    'type': 'File',
                    'outputSource': 'step1/result'
                }
            },
            'steps': [
                {
                    'id': 'step1',
                    'run': 'tool1.cwl',
                    'in': {
                        'input': 'input1'
                    },
                    'out': ['result']
                },
                {
                    'id': 'step2',
                    'run': 'tool2.cwl',
                    'in': {
                        'input': 'step1/result'
                    },
                    'out': ['result']
                }
            ]
        }
        
        # Create a workflow with steps as a dict
        workflow_dict = {
            'cwlVersion': 'v1.0',
            'class': 'Workflow',
            'inputs': {
                'input1': 'string'
            },
            'outputs': {
                'output1': {
                    'type': 'File',
                    'outputSource': 'step1/result'
                }
            },
            'steps': {
                'step1': {
                    'run': 'tool1.cwl',
                    'in': {
                        'input': 'input1'
                    },
                    'out': ['result']
                },
                'step2': {
                    'run': 'tool2.cwl',
                    'in': {
                        'input': 'step1/result'
                    },
                    'out': ['result']
                }
            }
        }
        
        # Write both workflows to temporary files
        workflow_list_path = tmp_path / "workflow_list.cwl"
        workflow_dict_path = tmp_path / "workflow_dict.cwl"
        
        with open(workflow_list_path, 'w') as f:
            yaml.dump(workflow_list, f)
        
        with open(workflow_dict_path, 'w') as f:
            yaml.dump(workflow_dict, f)
        
        # Load both workflows
        result_list = load_cwl_workflow(workflow_list_path)
        result_dict = load_cwl_workflow(workflow_dict_path)
        
        # Both should have the same structure
        assert 'steps' in result_list
        assert 'inputs' in result_list
        assert 'outputs' in result_list
        
        assert 'steps' in result_dict
        assert 'inputs' in result_dict
        assert 'outputs' in result_dict
        
        # Both should have steps as a list
        assert isinstance(result_list['steps'], list)
        assert isinstance(result_dict['steps'], list)
        
        # Both should have the same number of steps
        assert len(result_list['steps']) == 2
        assert len(result_dict['steps']) == 2
        
        # Both should have 'id' keys in their steps
        for step in result_list['steps']:
            assert 'id' in step
        
        for step in result_dict['steps']:
            assert 'id' in step
        
        # Sort steps by id for comparison
        steps_list_sorted = sorted(result_list['steps'], key=lambda s: s['id'])
        steps_dict_sorted = sorted(result_dict['steps'], key=lambda s: s['id'])
        
        # Verify they produce the same result
        assert steps_list_sorted == steps_dict_sorted
        
        # Verify inputs and outputs are preserved correctly
        assert result_list['inputs'] == result_dict['inputs']
        assert result_list['outputs'] == result_dict['outputs']


class TestMockedCWLExecution:
    """Tests for mocked_cwl_execution() function."""

    def test_mocked_execution_creates_json_output_file(self):
        """Test that mocked_cwl_execution creates JSON outputs file."""
        import json
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a simple CWL file
            cwl_file = Path(tmp_dir) / "workflow.cwl"
            cwl_content = dedent("""
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
            """)
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
            
            assert "output_file" in outputs
            assert "output_files" in outputs
            assert "output_dir" in outputs
            
            # Verify file output has class and path (matching real CWL format)
            assert isinstance(outputs["output_file"], dict)
            assert outputs["output_file"]["class"] == "File"
            assert Path(outputs["output_file"]["path"]).exists()
            
            # Verify array output is a list of File objects
            assert isinstance(outputs["output_files"], list)
            assert len(outputs["output_files"]) == 3
            for file_obj in outputs["output_files"]:
                assert file_obj["class"] == "File"
                assert "path" in file_obj
            
            # Verify directory output has class and path
            assert isinstance(outputs["output_dir"], dict)
            assert outputs["output_dir"]["class"] == "Directory"
            assert Path(outputs["output_dir"]["path"]).exists()
