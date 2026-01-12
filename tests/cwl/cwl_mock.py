
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
        'inputs': cwl_data.get('inputs'),
        'outputs': cwl_data.get('outputs')
    }


def _walk_array_type(array_object, suffix=""):
    """Recursively walk through nested array types to find the base type."""
    if isinstance(array_object, dict) and array_object.get("type") == "array":
        return _walk_array_type(array_object.get("items", {}), suffix=suffix + "[]")
    if isinstance(array_object, dict) and isinstance(array_object.get("type"), dict):
        return _walk_array_type(array_object.get("type", {}))
    if isinstance(array_object, dict) and isinstance(array_object.get("type"), str):
        return array_object["type"] + suffix
    if isinstance(array_object, str):
        return array_object + suffix
    return None

def get_output_type(output_info):
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
    output_type = output_info.get("type")
    
    # Handle simple types (string)
    if isinstance(output_type, str):
        return output_type.rstrip("?")
    if isinstance(output_type, list):
        first_not_null, *_ = [t for t in output_type if t != "null"]
        if isinstance(first_not_null, str):
            return first_not_null.rstrip("?")
        output_type = first_not_null
    # Handle dict types (array, object, etc.)
    return _walk_array_type(output_type)


def _create_mock_output(output_path, base_name, output_type, inner_size=3, outer_size=1):
    """Create a single mock output (file or directory).
    
    Args:
        output_path: Directory where outputs should be created
        base_name: Base name for the output
        output_type: Type of output (File, Directory, File[], etc.)
        n_files: Number of files/directories to create for array types
        outer_size: Size of the outer array for nested arrays (if applicable)
    """
    if output_type == "File":
        (output_path / base_name).touch()
    elif output_type == "Directory":
        (output_path / base_name).mkdir(parents=True, exist_ok=True)
    elif output_type == "File[]":
        for idx in range(inner_size):
            (output_path / f"{base_name}_{idx}").touch()
    elif output_type == "Directory[]":
        for idx in range(inner_size):
            (output_path / f"{base_name}_{idx}").mkdir(parents=True, exist_ok=True)
    elif output_type.endswith("[][]"):
        item_type = output_type.replace("[]", "", 1) # Remove one level of array
        for outer_idx in range(outer_size):
            _create_mock_output(output_path, f"{base_name}_list_{outer_idx}", item_type, inner_size=inner_size)

def _get_step_by_id(workflow, step_id):
    """Get a step from the workflow by its ID."""
    steps: list = workflow['steps']
    for step in steps:
        if step.get('id') == step_id:
            return step
    return None

def _get_size_of_input_array(inputs, param):
    """Get the size of an input array parameter."""
    if param in inputs:
        input_value = inputs[param]
        if isinstance(input_value, list):
            return len(input_value)
    return None

def _get_name_from_output_source(output_source):
    """Extract the name from an output source string."""
    if '/' in output_source:
        return output_source.split('/')[0]
    return output_source

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
    step_id = _get_name_from_output_source(output_source)
    
    # Find the step with matching ID
    target_step = _get_step_by_id(workflow_data, step_id)
    
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
            n_items_for_param  = _get_size_of_input_array(inputs, param)
            if n_items_for_param is not None:
                n_params += n_items_for_param
        if n_params > 0:
            return n_params
    # Default fallback
    return None


def generate_mock_files(output_path, outputs, mock_n_outer=3, mock_n_inner=3, workflow=None, inputs=None):
    """
    Generate mock output files/directories to mimic CWL execution.
    
    Supports: File, Directory, File[], Directory[], nested_list_*, nested_list_union_*
    
    Args:
        output_path: Directory where outputs should be created
        outputs: List of output specifications from CWL workflow
        mock_n_outer_files: Default number of outer files/directories to create for arrays (used if no scatter info)
        mock_n_inner_files: Default number of inner files/directories to create for nested arrays (used if no scatter info)
        workflow: Optional path to CWL workflow file for scatter information
        inputs: Optional dict of input values for determining scatter array lengths
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for output_info in outputs:
        output_type = get_output_type(output_info)
        if output_type is None:
            # Skip types that are not File or Directory
            # or their nested conterparts
            continue  
        output_sources = output_info.get('outputSource', [])
        if not isinstance(output_sources, list):
            output_sources = [output_sources]
        
        for output_source in output_sources:
            base_name = output_source.replace('/', '.')
            
            # Determine array length: use scatter info if available, otherwise default
            n_files = mock_n_outer
            if workflow and ("File" in output_type or "Directory" in output_type):
                scatter_length = get_scatter_length(workflow, output_source, inputs)
                if scatter_length is not None:
                    n_files = scatter_length
            
            _create_mock_output(output_path, base_name, output_type, inner_size=n_files, outer_size=mock_n_inner)


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
                for f in sorted(output_path.glob(f"{base_name}_*")) if f.is_file()]
    elif output_type == "Directory[]":
        return [{"class": "Directory", "path": str(d)} 
                for d in sorted(output_path.glob(f"{base_name}_*")) if d.is_dir()]
    elif output_type.endswith("[][]"):
        item_type = output_type.replace("[]", "") # Remove one level of array
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
        if output_type is None:
            # Skip types that are not File or Directory
            # or their nested conterparts
            continue
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
                print("output_type:", output_type)
                if workflow_file and ("File" in output_type or "Directory" in output_type):
                    scatter_length = get_scatter_length(workflow_file, output_source, inputs)
                    if scatter_length is not None:
                        n_files = scatter_length
                
                _create_mock_output(output_path, base_name, output_type, n_files)
                outputs_json[output_key] = _create_json_output(base_name, output_type, output_path)
    
    # Write JSON outputs file
    with open(self.operation.pipeline_outputs_file, 'w') as f:
        json.dump(outputs_json, f, indent=2)
    
    return True
