"""
Tests for CWL scatter-aware output generation.
"""
import pytest
import yaml

from ..cwl_mock import (
    load_cwl_workflow,
    infer_scatter_length,
    build_mock_outputs
)


class TestScatterAware:
    """Tests for scatter-aware output generation."""

    @pytest.fixture
    def simple_scatter_workflow(self, tmp_path):
        """Create a simple workflow with scatter."""
        workflow_file = tmp_path / "scatter_workflow.cwl"
        workflow_content = {
            "cwlVersion": "v1.2",
            "class": "Workflow",
            "inputs": {
                "input_files": {"type": {"type": "array", "items": "File"}},
                "input_dirs": {"type": {"type": "array", "items": "Directory"}}
            },
            "steps": [
                {
                    "id": "process_step",
                    "run": "some_tool.cwl",
                    "in": {
                        "files": {"source": "input_files"}
                    },
                    "scatter": ["files"],
                    "scatterMethod": "dotproduct",
                    "out": ["processed"]
                }
            ],
            "outputs": {
                "results": {
                    "type": {"type": "array", "items": "File"},
                    "outputSource": "process_step/processed"
                }
            }
        }
        with open(workflow_file, 'w') as f:
            yaml.dump(workflow_content, f)
        return workflow_file

    @pytest.fixture
    def nested_scatter_workflow(self, tmp_path):
        """Create a workflow with nested list outputs from scatter."""
        workflow_file = tmp_path / "nested_scatter_workflow.cwl"
        workflow_content = {
            "cwlVersion": "v1.2",
            "class": "Workflow",
            "inputs": {
                "observations": {"type": {"type": "array", "items": "Directory"}},
                "sectors": {"type": {"type": "array", "items": "File"}}
            },
            "steps": [
                {
                    "id": "predict_step",
                    "run": "predict.cwl",
                    "in": {
                        "obs": {"source": "observations"},
                        "sector": {"source": "sectors"}
                    },
                    "scatter": ["obs", "sector"],
                    "scatterMethod": "dotproduct",
                    "out": ["models"]
                },
                {
                    "id": "subtract_step",
                    "run": "subtract.cwl",
                    "in": {
                        "obs": {"source": "observations"},
                        "models": {"source": "predict_step/models"}
                    },
                    "scatter": ["obs"],
                    "out": ["subtracted"]
                }
            ],
            "outputs": {
                "output_models": {
                    "type": {
                        "type": "array",
                        "items": {"type": "array", "items": "Directory"}
                    },
                    "outputSource": "subtract_step/subtracted"
                }
            }
        }
        with open(workflow_file, 'w') as f:
            yaml.dump(workflow_content, f)
        return workflow_file

    def test_parse_cwl_workflow(self, simple_scatter_workflow):
        """Test parsing a CWL workflow file."""
        workflow_data = load_cwl_workflow(simple_scatter_workflow)
        
        assert 'steps' in workflow_data
        assert 'inputs' in workflow_data
        assert 'outputs' in workflow_data
        assert len(workflow_data['steps']) == 1
        assert workflow_data['steps'][0]['id'] == 'process_step'
        assert 'scatter' in workflow_data['steps'][0]

    def test_get_scatter_length_with_inputs(self, simple_scatter_workflow):
        """Test getting scatter length from workflow with inputs."""
        inputs = {
            "input_files": [
                {"class": "File", "path": "/path/to/file1.txt"},
                {"class": "File", "path": "/path/to/file2.txt"},
                {"class": "File", "path": "/path/to/file3.txt"},
                {"class": "File", "path": "/path/to/file4.txt"},
                {"class": "File", "path": "/path/to/file5.txt"}
            ]
        }
        
        # Simulate the scenario where scatter param "files" sources from "input_files"
        # In real scenarios, we'd need to trace the source chain
        length = infer_scatter_length(simple_scatter_workflow, "process_step/processed", 
                                    {"files": inputs["input_files"]})
        
        assert length == 5

    def test_get_scatter_length_no_scatter(self, tmp_path):
        """Test getting scatter length when step has no scatter."""
        workflow_file = tmp_path / "no_scatter.cwl"
        workflow_content = {
            "cwlVersion": "v1.2",
            "class": "Workflow",
            "inputs": {"input": {"type": "File"}},
            "steps": [
                {
                    "id": "simple_step",
                    "run": "tool.cwl",
                    "in": {"file": {"source": "input"}},
                    "out": ["output"]
                }
            ],
            "outputs": {
                "result": {
                    "type": "File",
                    "outputSource": "simple_step/output"
                }
            }
        }
        with open(workflow_file, 'w') as f:
            yaml.dump(workflow_content, f)
        
        length = infer_scatter_length(workflow_file, "simple_step/output", {})
        assert length is None

    def test_get_scatter_length_no_workflow(self):
        """Test getting scatter length when workflow is None."""
        length = infer_scatter_length(None, "step/output", {})
        assert length is None

    def test_generate_with_scatter_length(self, simple_scatter_workflow, tmp_path):
        """Test that generate_mock_files uses scatter length when available."""
        output_path = tmp_path / "outputs"
        
        # Define inputs with 5 elements
        inputs = {
            "files": [f"/path/to/file{i}.txt" for i in range(5)]
        }
        
        # Define outputs that come from scattered step
        outputs = {
            "processed": {
                "type": {"type": "array", "items": "File"},
                "outputSource": "process_step/processed"
            }
        }
        
        build_mock_outputs(output_path, outputs, 
                          workflow=simple_scatter_workflow, inputs=inputs)
        
        # Should create 5 files (matching scatter length) instead of default 3
        result = build_mock_outputs(output_path, outputs, 
                          workflow=simple_scatter_workflow, inputs=inputs)
        generated_files = result.get("processed", [])
        assert len(generated_files) == 5

    def test_generate_with_nested_scatter(self, nested_scatter_workflow, tmp_path):
        """Test generation of nested list outputs from scatter."""
        output_path = tmp_path / "outputs"
        
        # Define inputs
        inputs = {
            "obs": [f"/obs{i}" for i in range(4)],  # 4 observations
            "sector": [f"/sector{i}" for i in range(4)]  # 4 sectors
        }
        
        # Define nested list output
        outputs = {
            "subtracted": {
                "type": {
                    "type": "array",
                    "items": {"type": "array", "items": "Directory"}
                },
                "outputSource": "subtract_step/subtracted"
            }
        }
        
        build_mock_outputs(output_path, outputs,
                            workflow=nested_scatter_workflow, inputs=inputs)
        
        # Should create 4 outer directories (matching obs scatter length)
        result = build_mock_outputs(output_path, outputs,
                            workflow=nested_scatter_workflow, inputs=inputs)
        outer_list = result.get("subtracted", [])
        assert len(outer_list) == 4

    def test_fallback_to_default_without_inputs(self, simple_scatter_workflow, tmp_path):
        """Test that generation falls back to default when no inputs provided."""
        output_path = tmp_path / "outputs"
        
        outputs = {
            "processed": {
                "type": {"type": "array", "items": "File"},
                "outputSource": "process_step/processed"
            }
        }
        
        # Call without inputs - should use default mock_n_files
        build_mock_outputs(output_path, outputs, mock_n_outer=3,
                          workflow=simple_scatter_workflow, inputs=None)
        
        # Should create default 3 files
        result = build_mock_outputs(output_path, outputs, mock_n_outer=3,
                          workflow=simple_scatter_workflow, inputs=None)
        generated_files = result.get("processed", [])
        assert len(generated_files) == 3

    def test_mixed_scatter_and_non_scatter(self, tmp_path):
        """Test workflow with both scattered and non-scattered steps."""
        workflow_file = tmp_path / "mixed_workflow.cwl"
        workflow_content = {
            "cwlVersion": "v1.2",
            "class": "Workflow",
            "inputs": {
                "files": {"type": {"type": "array", "items": "File"}},
                "config": {"type": "File"}
            },
            "steps": [
                {
                    "id": "scattered_step",
                    "run": "tool1.cwl",
                    "in": {"input": {"source": "files"}},
                    "scatter": ["input"],
                    "out": ["output"]
                },
                {
                    "id": "regular_step",
                    "run": "tool2.cwl",
                    "in": {"config": {"source": "config"}},
                    "out": ["result"]
                }
            ],
            "outputs": {
                "scattered_results": {
                    "type": {"type": "array", "items": "File"},
                    "outputSource": "scattered_step/output"
                },
                "regular_result": {
                    "type": "File",
                    "outputSource": "regular_step/result"
                }
            }
        }
        with open(workflow_file, 'w') as f:
            yaml.dump(workflow_content, f)
        
        output_path = tmp_path / "outputs"
        inputs = {
            "input": [f"/file{i}" for i in range(7)]  # 7 files
        }
        
        outputs = {
            "scattered_results": {
                "type": {"type": "array", "items": "File"},
                "outputSource": "scattered_step/output"
            },
            "regular_result": {
                "type": "File",
                "outputSource": "regular_step/result"
            }
        }
        
        build_mock_outputs(output_path, outputs, workflow=workflow_file, inputs=inputs)
        
        # Scattered output should have 7 files
        result = build_mock_outputs(output_path, outputs, workflow=workflow_file, inputs=inputs)
        scattered_files = result.get("scattered_results", [])
        assert len(scattered_files) == 7
        
        # Regular output should be a file object
        regular_result = result.get("regular_result")
        assert regular_result is not None
        assert regular_result.get("class") == "File"
