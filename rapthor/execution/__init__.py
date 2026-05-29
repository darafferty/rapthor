"""Execution-layer helpers for the Prefect/Dask migration."""

from rapthor.execution.capabilities import (
    PreflightError,
    PreflightIssue,
    collect_preflight_issues,
    preflight_execution,
)
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.outputs import (
    OutputRecordError,
    directory_record,
    file_record,
    is_output_record,
    validate_output_record,
)

__all__ = [
    "ExecutionConfig",
    "OutputRecordError",
    "PreflightError",
    "PreflightIssue",
    "collect_preflight_issues",
    "directory_record",
    "file_record",
    "is_output_record",
    "preflight_execution",
    "validate_output_record",
]
