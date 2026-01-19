"""
Test to verify mock output structure matches real CWL execution.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from ..cwl_mock import build_mock_outputs
        

class TestRealCWLExecution:
    """Compare mock outputs with real CWL execution."""

    def test_compare_with_real_cwl_simple_tool(self):
        """Run a real CWL tool and verify output structure."""
        # Check if cwltool is available
        try:
            result = subprocess.run(['which', 'cwltool'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                pytest.skip("cwltool not installed")
        except Exception:
            pytest.skip("cwltool not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Run real CWL tool
            cwl_file = Path(__file__).parent / "test_data" / "real_cwl_test" / "simple_tool.cwl"
            
            result = subprocess.run(
                ['cwltool', '--outdir', str(tmpdir / "outputs"), str(cwl_file)],
                capture_output=True,
                text=True,
                cwd=tmpdir
            )
            
            # Parse the JSON output from cwltool
            # cwltool outputs JSON to stdout
            cwl_outputs = json.loads(result.stdout)
            
            # Verify structure matches what our mock produces
            # Single file should have class and path
            assert "single_file" in cwl_outputs
            assert isinstance(cwl_outputs["single_file"], dict)
            assert cwl_outputs["single_file"]["class"] == "File"
            assert "path" in cwl_outputs["single_file"]
            
            # Single directory should have class and path
            assert "single_dir" in cwl_outputs
            assert isinstance(cwl_outputs["single_dir"], dict)
            assert cwl_outputs["single_dir"]["class"] == "Directory"
            assert "path" in cwl_outputs["single_dir"]
            
            # File array should be list of dicts with class and path
            assert "file_array" in cwl_outputs
            assert isinstance(cwl_outputs["file_array"], list)
            for file_obj in cwl_outputs["file_array"]:
                assert isinstance(file_obj, dict)
                assert file_obj["class"] == "File"
                assert "path" in file_obj
            
            # Directory array should be list of dicts with class and path
            assert "dir_array" in cwl_outputs
            assert isinstance(cwl_outputs["dir_array"], list)
            for dir_obj in cwl_outputs["dir_array"]:
                assert isinstance(dir_obj, dict)
                assert dir_obj["class"] == "Directory"
                assert "path" in dir_obj

    def test_mock_matches_real_cwl_structure(self):
        """Verify our mock produces the same structure as real CWL."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Define outputs matching the CWL tool
            outputs = {
                "single_file": {"type": "File", "outputSource": "tool/single_file"},
                "single_dir": {"type": "Directory", "outputSource": "tool/single_dir"},
                "file_array": {
                    "type": {"type": "array", "items": "File"},
                    "outputSource": "tool/file_array"
                },
                "dir_array": {
                    "type": {"type": "array", "items": "Directory"},
                    "outputSource": "tool/dir_array"
                }
            }
            
            # Generate mock files
            output_path = tmpdir / "mock_outputs"
            mock_outputs = build_mock_outputs(output_path, outputs, mock_n_outer=3)
            
            # Verify mock structure matches CWL format
            # Single file
            assert "single_file" in mock_outputs
            assert isinstance(mock_outputs["single_file"], dict)
            assert mock_outputs["single_file"]["class"] == "File"
            assert "path" in mock_outputs["single_file"]
            
            # Single directory
            assert "single_dir" in mock_outputs
            assert isinstance(mock_outputs["single_dir"], dict)
            assert mock_outputs["single_dir"]["class"] == "Directory"
            assert "path" in mock_outputs["single_dir"]
            
            # File array
            assert "file_array" in mock_outputs
            assert isinstance(mock_outputs["file_array"], list)
            assert len(mock_outputs["file_array"]) == 3
            for file_obj in mock_outputs["file_array"]:
                assert isinstance(file_obj, dict)
                assert file_obj["class"] == "File"
                assert "path" in file_obj
            
            # Directory array
            assert "dir_array" in mock_outputs
            assert isinstance(mock_outputs["dir_array"], list)
            assert len(mock_outputs["dir_array"]) == 3
            for dir_obj in mock_outputs["dir_array"]:
                assert isinstance(dir_obj, dict)
                assert dir_obj["class"] == "Directory"
                assert "path" in dir_obj
