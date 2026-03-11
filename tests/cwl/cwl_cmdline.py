"""
Module for parsing CWL CommandLineTools and generating command lines.

Uses cwltool's built-in job generation to extract the exact command line
that would be executed.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from cwltool.command_line_tool import CommandLineTool
    from cwltool.context import LoadingContext, RuntimeContext
    from cwltool.job import CommandLineJob
    from cwltool.load_tool import load_tool
    from cwltool.process import Process
    from cwltool.workflow import default_make_tool
except ImportError:
    raise ImportError("cwltool is required for CWL parsing. Install with: pip install cwltool")


def load_cwl_tool(cwl_path: Union[str, Path]) -> "Process":
    """Load and parse a CWL CommandLineTool or Workflow using cwltool.

    Args:
        cwl_path: Path to the CWL file

    Returns:
        Parsed CWL tool/workflow object from cwltool
    """
    ctx = LoadingContext()
    ctx.construct_tool_object = default_make_tool
    return load_tool(str(Path(cwl_path).resolve()), ctx)


def generate_command_line(
    cwl_path: Union[str, Path],
    inputs: Optional[Dict[str, Any]] = None,
) -> Optional[List[str]]:
    """Generate the command line that would be executed for a CWL CommandLineTool.

    Uses cwltool's native job generation to produce the exact command line.

    Args:
        cwl_path: Path to the CWL CommandLineTool file
        inputs: Dict of input values (all required inputs must be provided)

    Returns:
        List of command line arguments

    Raises:
        ImportError: If cwltool is not installed
        ValueError: If the CWL file is not a CommandLineTool
        WorkflowException: If required inputs are missing

    Example:
        >>> cmd = generate_command_line("simple_tool.cwl", {"message": "Hello"})
        >>> print(" ".join(cmd))
        bash -c echo "Hello" > output.txt
    """
    tool = load_cwl_tool(cwl_path)

    if not isinstance(tool, CommandLineTool):  # type: ignore[arg-type]
        raise ValueError(
            f"Expected CommandLineTool, got {type(tool).__name__}. "
            "This function only supports CWL CommandLineTool files."
        )

    # Set up runtime context with temp directories
    rtctx = RuntimeContext()
    rtctx.outdir = tempfile.mkdtemp()
    rtctx.tmpdir = tempfile.mkdtemp()
    rtctx.stagedir = tempfile.mkdtemp()

    # Generate job and extract command line
    job_order = inputs or {}

    def output_callback(outputs: Any, process_status: str) -> None:
        """No-op callback for outputs."""
        pass

    for job in tool.job(job_order, output_callback, rtctx):
        if isinstance(job, CommandLineJob):
            return job.command_line

    return None
