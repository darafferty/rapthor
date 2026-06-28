"""Prefect flow for the Concatenate operation."""

from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.concatenate.commands import build_concatenate_command
from rapthor.execution.concatenate.payloads import (
    ConcatenateEpochPayload,
    validate_concatenate_payload,
)
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.outputs import require_directory
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.shell import run_external_command
from rapthor.execution.task_runner import run_flow_with_task_runner
from rapthor.lib.records import validate_output_record


def run_concatenate_epoch(
    epoch: ConcatenateEpochPayload,
    data_colname: str,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run concatenation for one epoch and return a directory output record."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    input_filenames = epoch["input_filenames"]
    output_filename = epoch["output_filename"]
    output_path = epoch["output_path"]
    command = build_concatenate_command(input_filenames, output_filename, data_colname)
    run_external_command(
        command,
        pipeline_working_dir,
        config,
        name="concatenate_epoch",
        shell_operation_cls=shell_operation_cls,
    )
    return require_directory(output_path, "Concatenate output")


@task(name="concatenate_epoch")
def concatenate_epoch_task(
    epoch: ConcatenateEpochPayload,
    data_colname: str,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one epoch concatenation."""
    with publish_python_logs_to_prefect():
        return run_concatenate_epoch(
            epoch,
            data_colname,
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )


def _result_from_epoch_records(outputs: list[dict]) -> dict:
    result = {"concatenated_filenames": outputs}
    validate_output_record(result["concatenated_filenames"])
    return result


def _run_concatenate_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    payload = validate_concatenate_payload(payload)
    outputs = [
        concatenate_epoch_task.submit(
            epoch,
            payload["data_colname"],
            payload["pipeline_working_dir"],
            execution_config=config,
        )
        for epoch in payload["epochs"]
    ]
    outputs = [output.result() for output in outputs]
    return _result_from_epoch_records(outputs)


@flow(name="concatenate")
def _concatenate_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect implementation for Concatenate."""
    with publish_python_logs_to_prefect():
        return _run_concatenate_prefect_tasks(
            payload,
            execution_config=execution_config,
        )


def concatenate_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect entry point for Concatenate."""
    return run_flow_with_task_runner(
        _concatenate_flow,
        payload,
        execution_config=execution_config,
    )
