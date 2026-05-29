"""Execution-layer helpers for the Prefect/Dask migration."""

from rapthor.execution.capabilities import (
    PreflightError,
    PreflightIssue,
    collect_preflight_issues,
    preflight_execution,
)
from rapthor.execution.commands import command_matches_fixture, command_to_string, normalize_command
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.concatenate import (
    build_concatenate_command,
    concatenate_epoch_task,
    concatenate_flow,
    concatenate_payload_from_inputs,
    normalized_concatenate_command,
    run_concatenate_epoch,
    run_concatenate_flow,
)
from rapthor.execution.flows.mosaic import (
    build_compress_mosaic_command,
    build_make_mosaic_command,
    build_make_mosaic_template_command,
    build_regrid_image_command,
    mosaic_flow,
    mosaic_image_type_task,
    mosaic_payload_from_inputs,
    normalized_make_mosaic_command,
    normalized_make_mosaic_template_command,
    normalized_regrid_image_command,
    run_mosaic_flow,
    run_mosaic_image_type,
)
from rapthor.execution.outputs import (
    OutputRecordError,
    directory_record,
    file_record,
    is_output_record,
    validate_output_record,
)
from rapthor.execution.payloads import PayloadSerializationError, assert_serializable_payload
from rapthor.execution.resources import ResourceRequest
from rapthor.execution.runtime import RuntimeSpec, UnsupportedRuntimeError, build_runtime_spec
from rapthor.execution.shell import MissingPrefectShellError, ShellCommand, run_shell_command
from rapthor.execution.task_runner import MissingPrefectDaskError, build_task_runner
from rapthor.execution.workdirs import WorkDirectoryLayout

__all__ = [
    "ExecutionConfig",
    "MissingPrefectDaskError",
    "MissingPrefectShellError",
    "OutputRecordError",
    "PayloadSerializationError",
    "PreflightError",
    "PreflightIssue",
    "ResourceRequest",
    "RuntimeSpec",
    "ShellCommand",
    "UnsupportedRuntimeError",
    "WorkDirectoryLayout",
    "assert_serializable_payload",
    "build_compress_mosaic_command",
    "build_concatenate_command",
    "build_make_mosaic_command",
    "build_make_mosaic_template_command",
    "build_regrid_image_command",
    "build_runtime_spec",
    "build_task_runner",
    "concatenate_epoch_task",
    "collect_preflight_issues",
    "command_matches_fixture",
    "command_to_string",
    "concatenate_flow",
    "concatenate_payload_from_inputs",
    "directory_record",
    "file_record",
    "is_output_record",
    "mosaic_flow",
    "mosaic_image_type_task",
    "mosaic_payload_from_inputs",
    "normalized_concatenate_command",
    "normalized_make_mosaic_command",
    "normalized_make_mosaic_template_command",
    "normalized_regrid_image_command",
    "normalize_command",
    "preflight_execution",
    "run_concatenate_epoch",
    "run_concatenate_flow",
    "run_mosaic_flow",
    "run_mosaic_image_type",
    "run_shell_command",
    "validate_output_record",
]
