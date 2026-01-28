"""
Module that contains utilities to mock the CWL execution in tests.
In particular, it can generate mock output files and directories
based on the CWL workflow outputs specification, including handling
of scatter directives to match output array lengths.
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml


def extract_cwl_outputs(workflow: Union[str, Path]) -> Dict[str, Any]:
    """Extract CWL workflow outputs from YAML file.
    
    Args:
        workflow: Path to CWL workflow file
    
    Returns:
        Dict of output specifications from the workflow
    """
    with open(workflow, 'r') as f:
        cwl_data = yaml.safe_load(f)
        outputs = cwl_data.get('outputs', {})
        if isinstance(outputs, list):
            # Convert list of outputs to dict format
            outputs_dict = {}
            for output in outputs:
                output_id = output.get('id', None)
                if output_id:
                    outputs_dict[output_id] = output
            return outputs_dict
    return outputs


def load_cwl_workflow(workflow: Union[str, Path]) -> Dict[str, Any]:
    """Load CWL workflow to get steps and scatter information.
    
    Args:
        workflow: Path to CWL workflow file
    
    Returns:
        Dict with 'steps', 'inputs', and 'outputs' keys from the workflow
    """
    with open(workflow, 'r') as f:
        cwl_data = yaml.safe_load(f)
    
    steps = cwl_data.get('steps', [])
    # Convert dict steps to list format if needed
    if isinstance(steps, dict):
        steps = [{"id": key, **value} for key, value in steps.items()]
    
    return {
        'steps': steps,
        'inputs': cwl_data.get('inputs'),
        'outputs': cwl_data.get('outputs')
    }


def _resolve_array_base_type(array_object: Union[Dict[str, Any], str], suffix: str = "") -> Optional[str]:
    """Recursively walk through nested array types to find the base type.
    
    Args:
        array_object: CWL type specification (dict or string)
        suffix: Accumulator for array brackets (default: "")
    
    Returns:
        Base type with accumulated array brackets (e.g., "File[][]")
    """
    if isinstance(array_object, dict) and array_object.get("type") == "array":
        return _resolve_array_base_type(array_object.get("items", {}), suffix=suffix + "[]")
    if isinstance(array_object, dict) and isinstance(array_object.get("type"), dict):
        return _resolve_array_base_type(array_object.get("type", {}))
    if isinstance(array_object, dict) and isinstance(array_object.get("type"), str):
        return array_object["type"] + suffix
    if isinstance(array_object, str):
        return array_object + suffix
    return None

def infer_output_type(output_info: Union[Dict[str, Any], None]) -> Optional[str]:
    """
    Determine the output type from CWL output specification.
    
    Returns one of:
    - "File"
    - "Directory"
    - "File[]"
    - "Directory[]"
    - "File[][]" (nested list)
    - "Directory[][]" (nested list)
    - None (if not a file/directory type)
    """
    if output_info is None:
        return None
    output_type = output_info.get("type", None)
    if output_type is None:
        return None
    # Handle simple types (string)
    if isinstance(output_type, str):
        return output_type.rstrip("?")
    if isinstance(output_type, list):
        first_not_null, *_ = [t for t in output_type if t != "null"]
        if isinstance(first_not_null, str):
            return first_not_null.rstrip("?")
        output_type = first_not_null
    # Handle dict types (array, object, etc.)
    return _resolve_array_base_type(output_type)



def _find_step(workflow: Dict[str, Any], step_id: str) -> Optional[Dict[str, Any]]:
    """Get a step from the workflow by its ID.
    
    Args:
        workflow: Parsed CWL workflow dict
        step_id: ID of the step to find
    
    Returns:
        Step specification dict, or None if not found
    """
    steps: list = workflow['steps']
    for step in steps:
        if step.get('id') == step_id:
            return step
    return None

def _input_array_len(inputs: Dict[str, Any], param: str) -> int:
    """Get the size of an input array parameter.
    
    Args:
        inputs: Dict of input values
        param: Name of the parameter to check
    
    Returns:
        Length of the array if parameter is a list, 0 otherwise
    """
    if param in inputs:
        input_value = inputs[param]
        if isinstance(input_value, list):
            return len(input_value)
    return 0

def _step_name_from_source(output_source: str) -> str:
    """Extract the step name from an output source string.
    
    Args:
        output_source: Output source string (e.g., 'step_name/output_name')
    
    Returns:
        Step name extracted from the output source
    """
    if '/' in output_source:
        return output_source.split('/')[0]
    return output_source

def infer_scatter_length(workflow: Optional[Union[str, Path, Dict[str, Any]]], output_source: str, inputs: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Get the length of scatter array for a given output source.
    
    Args:
        workflow: Path to CWL workflow file or parsed workflow dict (or None)
        output_source: Output source string (e.g., 'step_name/output_name')
        inputs: Optional dict of input values to determine actual array lengths
    
    Returns:
        Integer length of scatter array, or None if not scattered or workflow is None
    """
    if workflow is None:
        return None
    
    # Parse workflow if it's a file path
    if isinstance(workflow, (str, Path)):
        workflow_data = load_cwl_workflow(workflow)
    else:
        workflow_data = workflow
    
    # Extract step name from output source
    step_id = _step_name_from_source(output_source)
    
    # Find the step with matching ID
    target_step = _find_step(workflow_data, step_id)
    
    if target_step is None or 'scatter' not in target_step:
        return None
    
    # Get the scatter parameter(s)
    scatter_params = target_step['scatter']
    if not isinstance(scatter_params, list):
        scatter_params = [scatter_params]
    
    # If we have inputs, determine actual length
    if inputs and scatter_params:
        # Find the first scatter parameter and get its length
        n_params = 0
        for param in scatter_params:
            n_items_for_param  = _input_array_len(inputs, param)
            
            n_params = max(n_params, n_items_for_param)
        if n_params > 0:
            return n_params
    # Default fallback
    return None

def _build_mock_output(output_path: Path,
                       base_name: str,
                       output_type: str,
                       outer_size: int = 1,
                       inner_size: int = 3,) -> Union[Dict[str, Any], List[Any], None]:
    """Create a single mock output (file or directory).
    
    Args:
        output_path: Directory where outputs should be created
        base_name: Base name for the output
        output_type: Type of output (File, Directory, File[], Directory[], File[][], Directory[][])
        inner_size: Number of items for the inner array dimension (for nested arrays)
        outer_size: Number of items for the outer array dimension (for simple and nested arrays)
    
    Returns:
        Dict with CWL format (class and path), or list of dicts for arrays, or nested lists for nested arrays
    """
    if output_type in ["File", "Directory"]:
        return _create_path_entry(output_path, base_name, is_dir=(output_type == "Directory"))
    elif output_type == "File[]":
        return [_build_mock_output(output_path, f"{base_name}_{idx}", "File") for idx in range(outer_size)]
    elif output_type == "Directory[]":
        return [_build_mock_output(output_path, f"{base_name}_{idx}", "Directory") for idx in range(outer_size)]
    elif output_type.endswith("[][]"):
        # For nested arrays, outer_size controls the outer dimension; inner_size controls the inner dimension
        item_type = output_type.replace("[]", "", 1)
        return [
            _build_mock_output(output_path, f"{base_name}_list_{outer_idx}", item_type, outer_size=inner_size, inner_size=-1)
            for outer_idx in range(outer_size)
        ]
    return None

def _resolve_outer_length(workflow: Optional[Union[str, Path, Dict[str, Any]]], output_source: str, inputs: Optional[Dict[str, Any]], default_outer_size: int) -> int:
    """Determine the outer array length based on scatter information.
    
    If the workflow has scatter directives and inputs are provided, returns the scatter length.
    Otherwise returns the default outer size.
    
    Args:
        workflow: Path to CWL workflow file or parsed workflow dict
        output_source: Output source string (e.g., 'step_name/output_name')
        inputs: Dict of input values for determining scatter array lengths
        default_outer_size: Default size to use if no scatter info is found
    
    Returns:
        Integer length for the outer array dimension
    """
    scatter_length = infer_scatter_length(workflow, output_source, inputs)
    if scatter_length is not None:
        return scatter_length
    return default_outer_size


def _materialize_expected_outputs(output_path: Path, expected_filenames: Union[str, List[Any]]) -> Union[Dict[str, str], List[Any]]:
    """Generate output files/directories from expected filenames with proper structure.
    
    Args:
        output_path: Base directory where files/directories should be created
        expected_filenames: Expected filenames which can be:
                          - A string (single file/directory)
                          - A list of strings (simple array)
                          - A list of lists (nested array)
    
    Returns:
        Properly structured output matching the input structure
    """
    if isinstance(expected_filenames, list) and len(expected_filenames) > 0 and isinstance(expected_filenames[0], list):
        # Nested list - array of arrays
        nested_list = []
        for inner_filenames in expected_filenames:
            inner_list = [_create_path_entry(output_path, fn) for fn in inner_filenames]
            nested_list.append(inner_list)
        return nested_list
    elif isinstance(expected_filenames, list):
        # Simple array
        return [_create_path_entry(output_path, fn) for fn in expected_filenames]
    else:
        # Single file/directory
        return _create_path_entry(output_path, expected_filenames)

def _build_outputs_for_sources(output_path: Path,
                               output_info: Dict[str, Any],
                               mock_n_outer: int = 3,
                               mock_n_inner: int = 3, 
                               workflow: Optional[Union[str, Path, Dict[str, Any]]] = None,
                               inputs: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], List[Any], None]:
    """Generate mock output files/directories for a single output specification.
    
    Handles scatter-aware output generation: if the workflow has scatter directives and inputs
    are provided, the output array lengths will match the scatter input lengths.
    
    Supports: File, Directory, File[], Directory[], File[][], Directory[][]
    
    Args:
        output_path: Directory where outputs should be created
        output_info: Output specification from CWL workflow
        mock_n_outer: Default number of items for outer array dimension (used if no scatter info)
        mock_n_inner: Default number of items for inner array dimension (for nested arrays)
        workflow: Optional path to CWL workflow file for scatter information
        inputs: Optional dict of input values for determining scatter array lengths
    
    Returns:
        Generated output structure matching the output type (single dict, list, or nested list)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_summary: Union[Dict[str, Any], List[Any], None] = []
    output_type = infer_output_type(output_info)
    output_sources = output_info.get('outputSource', [])
    if output_type is None:
        # Skip types that are not File or Directory
        # or their nested conterparts
        output_summary.append(",".join(output_sources))
        return output_summary
    
    if not isinstance(output_sources, list):
        base_name = output_sources.replace('/', '.')
        if workflow and ("[]" in output_type):
            mock_n_outer = _resolve_outer_length(workflow, output_sources, inputs, mock_n_outer)
        output_summary = _build_mock_output(output_path, base_name, output_type, inner_size=mock_n_inner, outer_size=mock_n_outer)
        return output_summary
    
    for output_source in output_sources:
        base_name = output_source.replace('/', '.')
        
        # Determine array length: use scatter info if available, otherwise default
        mock_n_outer = _resolve_outer_length(workflow, output_source, inputs, mock_n_outer)
        mock_output = _build_mock_output(output_path, base_name, output_type, inner_size=mock_n_inner, outer_size=mock_n_outer)
        if isinstance(mock_output, list):
            output_summary.extend(mock_output)
        else:
            output_summary.append(mock_output)

    return output_summary


def build_mock_outputs(output_path: Path, outputs: Dict[str, Any], expected_outputs: Optional[Dict[str, Any]] = None, mock_n_outer: int = 3, mock_n_inner: int = 3, workflow: Optional[Union[str, Path]] = None, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate mock output files/directories to mimic CWL execution.
    
    Handles scatter-aware output generation: if the workflow has scatter directives and inputs
    are provided, the output array lengths will match the scatter input lengths.
    
    Supports: File, Directory, File[], Directory[], File[][], Directory[][]
    
    Args:
        output_path: Directory where outputs should be created
        outputs: Dict of output specifications from CWL workflow
        expected_outputs: Optional dict mapping output key to filename overrides.
                         Can override filenames at the first nested level only.
        mock_n_outer: Default number of items for outer array dimension (used if no scatter info)
        mock_n_inner: Default number of items for inner array dimension (for nested arrays)
        workflow: Optional path to CWL workflow file for scatter information
        inputs: Optional dict of input values for determining scatter array lengths
    
    Returns:
        Dict mapping output key to generated output structure (matching CWL format)
    """
    if expected_outputs is None:
        expected_outputs = {}
    
    outputs_json = {}
    
    for output_key, output_info in outputs.items():
        output_type = infer_output_type(output_info)
        if output_type is None:
            # Skip types that are not File or Directory
            # or their nested conterparts
            continue
        output_sources = output_info.get('outputSource', [])
        if not isinstance(output_sources, list):
            output_sources = [output_sources]
            
        # Check if this output has filename overrides
        if expected_outputs and output_key in expected_outputs:
            for output_source in output_sources:
                base_name = output_source.replace('/', '.')
                output_key = base_name.split(".")[-1]
                filenames = expected_outputs[output_key]
                outputs_json[output_key] = _materialize_expected_outputs(output_path, filenames)
        else:
            # No override, generate with auto names based on CWL output type
            # Determine array length from scatter if available
            outputs_json[output_key] = _build_outputs_for_sources(output_path,
                                output_info,
                                mock_n_inner=mock_n_inner,
                                mock_n_outer=mock_n_outer,
                                workflow=workflow,
                                inputs=inputs)   
    return outputs_json


def _create_path_entry(output_path: Path, filename: str, is_dir: bool = False) -> Dict[str, str]:
    """Create a single file or directory with CWL-compliant output structure.

    For JSON files, creates a minimal valid JSON content (empty object).
    Directories are inferred when is_dir=True or filename ends with '/' or has no extension.

    Args:
        output_path: Base directory where the file/directory should be created
        filename: Name of the file or directory to create
        is_dir: If True, create as directory; otherwise infer from filename

    Returns:
        Dict with "class" (File|Directory) and "path" fields matching CWL format
    """
    file_path = output_path / filename
    if is_dir or filename.endswith('/'):
        # Directory
        file_path.mkdir(parents=True, exist_ok=True)
        return {"class": "Directory", "path": str(file_path)}
    else:
        # File
        # Create empty file or JSON file with minimal content if it's a JSON file
        if filename.endswith('.json'):
            with open(file_path, 'w') as f:
                json.dump({}, f)  # Empty JSON object
        else:
            file_path.touch()
        return {"class": "File", "path": str(file_path)}


def mock_cwl_execution(expected_outputs: Optional[Dict[str, Any]]) -> Callable:
    """Decorator to mock CWL execution in operation tests.
    
    Replaces the operation's cwl_execute method with mocked execution that generates
    mock output files and outputs JSON file.
    
    Args:
        expected_outputs: Optional dict mapping output key to filename overrides
    
    Returns:
        Decorator function that wraps test methods that expect self.operation
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Replace the operation's CWL execution method with the mock
            self.operation.cwl_execute = lambda args, env: mocked_cwl_execution(self, args, env, expected_outputs=expected_outputs)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator
 

def mocked_cwl_execution(self: Any, args: Any, env: Any, expected_outputs: Optional[Dict[str, Any]] = None) -> bool:
    """Mock CWL execution by generating output files and outputs JSON.
    
    Supports scatter-aware output generation: if the workflow has scatter directives
    and inputs are provided, the output array lengths will match the scatter input lengths.
    
    Args:
        self: Test instance with operation attribute
        args: CWL command arguments (unused in mock)
        env: Environment variables (unused in mock)
        expected_outputs: Optional dict mapping output key to filename overrides.
                         The output structure is determined by CWL workflow,
                         expected_outputs only overrides the filenames at the first nested level.
    
    Returns:
        True on successful mock execution
    """
    output_path = Path(self.operation.parset['dir_working'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse workflow and outputs from CWL
    workflow_file = self.operation.pipeline_parset_file
    outputs = extract_cwl_outputs(workflow_file)
    
    # Try to get inputs if available (for scatter length determination)
    inputs = getattr(self.operation, 'input_parms', None)
    outputs_json = build_mock_outputs(output_path, outputs,
                        expected_outputs=expected_outputs,
                        workflow=workflow_file,
                        inputs=inputs)
    # Write JSON outputs file
    with open(self.operation.pipeline_outputs_file, 'w') as f:
        json.dump(outputs_json, f, indent=2)
    
    return True
