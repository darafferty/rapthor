
import json
from pathlib import Path

import yaml


def parse_cwl_outputs(workflow):
    """Parse CWL workflow outputs from YAML file."""
    with open(workflow, 'r') as f:
        cwl_data = yaml.safe_load(f)
        outputs = cwl_data.get('outputs', {})
    
    # Convert dict outputs to list format
    if isinstance(outputs, dict):
        return [output_spec if isinstance(output_spec, dict) else {"type": output_spec}
                for output_spec in outputs.values()]
    return outputs


def parse_cwl_workflow(workflow):
    """Parse CWL workflow to get steps and scatter information."""
    with open(workflow, 'r') as f:
        cwl_data = yaml.safe_load(f)
    
    steps = cwl_data.get('steps', [])
    # Convert dict steps to list format if needed
    if isinstance(steps, dict):
        steps = list(steps.values())
    
    return {
        'steps': steps,
        'inputs': cwl_data.get('inputs', {}),
        'outputs': cwl_data.get('outputs', {})
    }


def _extract_item_type(items_type):
    """Extract item type from items specification."""
    if isinstance(items_type, dict):
        return items_type.get("type", "File")
    elif isinstance(items_type, str):
        return items_type
    return "File"


def _get_array_type(items_type):
    """Get array type from items specification."""
    return f"{_extract_item_type(items_type)}[]"


def get_output_type(output_info):
    """
    Determine the output type from CWL output specification.
    
    Returns one of: 'File', 'Directory', 'File[]', 'Directory[]', 'nested_list_File', 
    'nested_list_Directory', or 'nested_list_union'
    """
    output_type = output_info.get("type")
    
    # Handle simple types (string)
    if isinstance(output_type, str):
        return output_type
    
    # Handle dict types (array, object, etc.)
    if isinstance(output_type, dict) and output_type.get("type") == "array":
        items_type = output_type.get("items")
        
        # Nested array: array of arrays
        if isinstance(items_type, dict) and items_type.get("type") == "array":
            inner_type = _extract_item_type(items_type.get("items", {}))
            return f"nested_list_{inner_type}"
        
        # Union with array: array of [array|null]
        if isinstance(items_type, list):
            for item in items_type:
                if isinstance(item, dict) and item.get("type") == "array":
                    inner_type = _extract_item_type(item.get("items", {}))
                    return f"nested_list_union_{inner_type}"
        
        # Simple array
        return _get_array_type(items_type)
    
    # Handle union types as list [type, "null"]
    if isinstance(output_type, list):
        for t in output_type:
            if t != "null":
                if isinstance(t, str):
                    return t
                if isinstance(t, dict) and t.get("type") == "array":
                    return _get_array_type(t.get("items", {}))
    
    return "File"


def _create_mock_output(output_path, base_name, output_type, n_files=3):
    """Create a single mock output (file or directory).
    
    Args:
        output_path: Directory where outputs should be created
        base_name: Base name for the output
        output_type: Type of output (File, Directory, File[], etc.)
        n_files: Number of files/directories to create for array types
    """
    if output_type == "File":
        (output_path / base_name).touch()
    elif output_type == "Directory":
        (output_path / base_name).mkdir(parents=True, exist_ok=True)
    elif output_type == "File[]":
        for idx in range(n_files):
            (output_path / f"{base_name}_{idx}.fits").touch()
    elif output_type == "Directory[]":
        for idx in range(n_files):
            (output_path / f"{base_name}_{idx}").mkdir(parents=True, exist_ok=True)
    elif output_type.startswith("nested_list"):
        item_type = output_type.replace("nested_list_", "").replace("union_", "")
        for outer_idx in range(n_files):
            outer_dir = output_path / f"{base_name}_list_{outer_idx}"
            outer_dir.mkdir(parents=True, exist_ok=True)
            # For nested lists, inner size is 2 by default (can be made configurable if needed)
            for inner_idx in range(2):
                if item_type == "File":
                    (outer_dir / f"item_{inner_idx}.fits").touch()
                elif item_type == "Directory":
                    (outer_dir / f"item_{inner_idx}").mkdir(parents=True, exist_ok=True)


def get_scatter_length(workflow, output_source, inputs=None):
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
        workflow_data = parse_cwl_workflow(workflow)
    else:
        workflow_data = workflow
    
    # Extract step name from output source
    if '/' in output_source:
        step_id = output_source.split('/')[0]
    else:
        return None
    
    # Find the step with matching ID
    steps = workflow_data.get('steps', [])
    target_step = None
    for step in steps:
        if step.get('id') == step_id:
            target_step = step
            break
    
    if not target_step or 'scatter' not in target_step:
        return None
    
    # Get the scatter parameter(s)
    scatter_params = target_step['scatter']
    if not isinstance(scatter_params, list):
        scatter_params = [scatter_params]
    
    # If we have inputs, determine actual length
    if inputs and len(scatter_params) > 0:
        # Find the first scatter parameter and get its length
        for param in scatter_params:
            if param in inputs:
                input_value = inputs[param]
                if isinstance(input_value, list):
                    return len(input_value)
    
    # Default fallback
    return None


def generate_mock_files(output_path, outputs, mock_n_files=3, workflow=None, inputs=None):
    """
    Generate mock output files/directories to mimic CWL execution.
    
    Supports: File, Directory, File[], Directory[], nested_list_*, nested_list_union_*
    
    Args:
        output_path: Directory where outputs should be created
        outputs: List of output specifications from CWL workflow
        mock_n_files: Default number of files to create for arrays (used if no scatter info)
        workflow: Optional path to CWL workflow file for scatter information
        inputs: Optional dict of input values for determining scatter array lengths
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for output_info in outputs:
        output_type = get_output_type(output_info)
        output_sources = output_info.get('outputSource', [])
        if not isinstance(output_sources, list):
            output_sources = [output_sources]
        
        for output_source in output_sources:
            base_name = output_source.replace('/', '.')
            
            # Determine array length: use scatter info if available, otherwise default
            n_files = mock_n_files
            if workflow and output_type in ['File[]', 'Directory[]'] or output_type.startswith('nested_list'):
                scatter_length = get_scatter_length(workflow, output_source, inputs)
                if scatter_length is not None:
                    n_files = scatter_length
            
            _create_mock_output(output_path, base_name, output_type, n_files)


def _create_json_output(base_name, output_type, output_path):
    """Create JSON representation of an output.
    
    Args:
        base_name: Base name of the output
        output_type: Type of output (File, Directory, File[], etc.)
        output_path: Directory where outputs were created
    
    Returns:
        JSON-serializable representation of the output matching CWL format
    """
    if output_type == "File":
        return {"class": "File", "path": str(output_path / base_name)}
    elif output_type == "Directory":
        return {"class": "Directory", "path": str(output_path / base_name)}
    elif output_type == "File[]":
        return [{"class": "File", "path": str(f)} 
                for f in sorted(output_path.glob(f"{base_name}_*.fits"))]
    elif output_type == "Directory[]":
        return [{"class": "Directory", "path": str(d)} 
                for d in sorted(output_path.glob(f"{base_name}_*")) if d.is_dir()]
    elif output_type.startswith("nested_list"):
        item_type = output_type.replace("nested_list_", "").replace("union_", "")
        nested_list = []
        for outer_dir in sorted(output_path.glob(f"{base_name}_list_*")):
            if outer_dir.is_dir():
                inner_list = []
                for inner_item in sorted(outer_dir.iterdir()):
                    if item_type == "File":
                        inner_list.append({"class": "File", "path": str(inner_item)})
                    elif item_type == "Directory":
                        inner_list.append({"class": "Directory", "path": str(inner_item)})
                nested_list.append(inner_list)
        return nested_list
    return None

def _create_file_or_directory(output_path, filename):
    """Create a single file or directory and return its JSON representation.
    
    Args:
        output_path: Base path where the file/directory should be created
        filename: Name of the file or directory to create
    
    Returns:
        Dict with "class" and "path" fields matching CWL format
    """
    file_path = output_path / filename
    if filename.endswith('/') or '.' not in filename.split('/')[-1]:
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


def generate_cwl_mocked_execution(expected_outputs):
    """Decorator to mock CWL execution in operation tests.
    
    Args:
        expected_outputs: List of output specifications from CWL workflow
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Replace the operation's CWL execution method with the mock
            self.operation.cwl_execute = lambda args, env: mocked_cwl_execution(self, args, env, expected_outputs=expected_outputs)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def mocked_cwl_execution(self, args, env, expected_outputs=None):
    """Mock CWL execution by generating output files and outputs JSON.
    
    Supports scatter-aware output generation: if the workflow has scatter directives
    and inputs are provided, the output array lengths will match the scatter input lengths.
    
    Args:
        expected_outputs: Optional dict mapping output key to filename overrides.
                         The output structure is determined by CWL workflow,
                         expected_outputs only overrides the filenames at the first nested level.
    """
    output_path = Path(self.operation.parset['dir_working'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse workflow and outputs from CWL
    workflow_file = self.operation.pipeline_parset_file
    outputs = parse_cwl_outputs(workflow_file)
    
    # Try to get inputs if available (for scatter length determination)
    inputs = getattr(self.operation, 'inputs', None)
    
    outputs_json = {}
    
    for output_info in outputs:
        output_type = get_output_type(output_info)
        output_sources = output_info.get('outputSource', [])
        if not isinstance(output_sources, list):
            output_sources = [output_sources]
        
        for output_source in output_sources:
            base_name = output_source.replace('/', '.')
            output_key = base_name.split(".")[-1]
            
            # Check if this output has filename overrides
            if expected_outputs and output_key in expected_outputs:
                filenames = expected_outputs[output_key]
                
                # Generate files with custom filenames based on structure
                if isinstance(filenames, list) and len(filenames) > 0 and isinstance(filenames[0], list):
                    # Nested list - array of arrays
                    nested_list = []
                    for inner_filenames in filenames:
                        inner_list = [_create_file_or_directory(output_path, fn) for fn in inner_filenames]
                        nested_list.append(inner_list)
                    outputs_json[output_key] = nested_list
                elif isinstance(filenames, list):
                    # Simple array
                    outputs_json[output_key] = [_create_file_or_directory(output_path, fn) for fn in filenames]
                else:
                    # Single file/directory
                    outputs_json[output_key] = _create_file_or_directory(output_path, filenames)
            else:
                # No override, generate with auto names based on CWL output type
                # Determine array length from scatter if available
                n_files = 3  # default
                if workflow_file and (output_type in ['File[]', 'Directory[]'] or output_type.startswith('nested_list')):
                    scatter_length = get_scatter_length(workflow_file, output_source, inputs)
                    if scatter_length is not None:
                        n_files = scatter_length
                
                _create_mock_output(output_path, base_name, output_type, n_files)
                outputs_json[output_key] = _create_json_output(base_name, output_type, output_path)
    
    # Write JSON outputs file
    with open(self.operation.pipeline_outputs_file, 'w') as f:
        json.dump(outputs_json, f, indent=2)
    
    return True
