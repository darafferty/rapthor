"""Prefect flow for the Concatenate operation."""

import os
from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.commands import normalize_command
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.runtime import run_flow_with_task_runner
from rapthor.execution.outputs import directory_record, validate_output_record
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.shell import ShellCommand, run_shell_command


def _record_path(record: object) -> str:
    if isinstance(record, Mapping) and record.get("class") == "Directory":
        path = record.get("path")
        if isinstance(path, str) and path:
            return path
    raise ValueError(f"Expected a Directory output record, got {record!r}")


def _validate_output_filename(output_filename: object, index: int) -> str:
    if not isinstance(output_filename, str) or not output_filename:
        raise ValueError(f"output_filenames[{index}] must be a non-empty string")
    if os.path.isabs(output_filename) or os.path.basename(output_filename) != output_filename:
        raise ValueError(f"output_filenames[{index}] must be a basename")
    return output_filename


def _validate_epoch_payloads(
    payload: Mapping[str, object],
) -> tuple[str, str, list[Mapping]]:
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    data_colname = str(payload["data_colname"])
    epochs = payload.get("epochs", [])
    if not isinstance(epochs, list):
        raise ValueError("epochs must be a list")
    for index, epoch in enumerate(epochs):
        if not isinstance(epoch, Mapping):
            raise ValueError(f"epochs[{index}] must be a mapping")
        output_filename = _validate_output_filename(epoch.get("output_filename"), index)
        expected_output_path = os.path.join(pipeline_working_dir, output_filename)
        if str(epoch.get("output_path")) != expected_output_path:
            raise ValueError(f"epochs[{index}].output_path must be {expected_output_path}")
    return pipeline_working_dir, data_colname, epochs


def _validate_unique_output_paths(epochs: list[Mapping]) -> None:
    seen_output_paths = set()
    for index, epoch in enumerate(epochs):
        output_path = str(epoch["output_path"])
        if output_path in seen_output_paths:
            raise ValueError(f"epochs[{index}].output_path must be unique")
        seen_output_paths.add(output_path)


def build_concatenate_command(
    input_filenames: list[str],
    output_filename: str,
    data_colname: str,
) -> list[str]:
    """Build the `concat_ms.py` command for one epoch."""
    return [
        "concat_ms.py",
        *input_filenames,
        f"--msout={output_filename}",
        "--concat_property=frequency",
        f"--data_colname={data_colname}",
    ]


def concatenate_payload_from_inputs(
    input_parms: Mapping[str, object],
    pipeline_working_dir: object,
) -> dict:
    """Create a serializable Concatenate flow payload from operation inputs."""
    pipeline_dir = str(pipeline_working_dir)
    input_filenames = input_parms.get("input_filenames", [])
    output_filenames = input_parms.get("output_filenames", [])
    data_colname = input_parms.get("data_colname", "DATA")

    if not isinstance(input_filenames, list):
        raise ValueError("input_filenames must be a list")
    if not isinstance(output_filenames, list):
        raise ValueError("output_filenames must be a list")
    if len(input_filenames) != len(output_filenames):
        raise ValueError("input_filenames and output_filenames must have the same length")
    if not isinstance(data_colname, str):
        raise ValueError("data_colname must be a string")

    epochs = []
    for index, (epoch_inputs, output_filename) in enumerate(zip(input_filenames, output_filenames)):
        if not isinstance(epoch_inputs, list):
            raise ValueError(f"input_filenames[{index}] must be a list")
        output_filename = _validate_output_filename(output_filename, index)
        epochs.append(
            {
                "input_filenames": [_record_path(record) for record in epoch_inputs],
                "output_filename": output_filename,
                "output_path": os.path.join(pipeline_dir, output_filename),
            }
        )

    payload = {
        "pipeline_working_dir": pipeline_dir,
        "data_colname": data_colname,
        "epochs": epochs,
    }
    _validate_unique_output_paths(epochs)
    return assert_serializable_payload(payload)


def run_concatenate_epoch(
    epoch: Mapping[str, object],
    data_colname: str,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run concatenation for one epoch and return a directory output record."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    input_filenames = list(epoch["input_filenames"])
    output_filename = str(epoch["output_filename"])
    output_path = str(epoch["output_path"])
    command = build_concatenate_command(input_filenames, output_filename, data_colname)
    run_shell_command(
        ShellCommand(
            command=command,
            working_directory=pipeline_working_dir,
            name="concatenate_epoch",
        ),
        config,
        shell_operation_cls=shell_operation_cls,
    )
    if not os.path.isdir(output_path):
        raise FileNotFoundError(f"Concatenate output was not created: {output_path}")
    return directory_record(output_path)


@task(name="concatenate_epoch")
def concatenate_epoch_task(
    epoch: Mapping[str, object],
    data_colname: str,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one epoch concatenation."""
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


def run_concatenate_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run concatenation commands and return finalizer-compatible outputs."""
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    pipeline_working_dir, data_colname, epochs = _validate_epoch_payloads(payload)
    _validate_unique_output_paths(epochs)
    outputs = []

    for epoch in epochs:
        outputs.append(
            run_concatenate_epoch(
                epoch,
                data_colname,
                pipeline_working_dir,
                execution_config=config,
                shell_operation_cls=shell_operation_cls,
            )
        )

    return _result_from_epoch_records(outputs)


def _run_concatenate_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    config = execution_config or ExecutionConfig(task_runner="sync")
    pipeline_working_dir, data_colname, epochs = _validate_epoch_payloads(payload)
    _validate_unique_output_paths(epochs)
    outputs = [
        concatenate_epoch_task.submit(
            epoch,
            data_colname,
            pipeline_working_dir,
            execution_config=config,
        )
        for epoch in epochs
    ]
    outputs = [output.result() for output in outputs]
    return _result_from_epoch_records(outputs)


@flow(name="concatenate")
def _concatenate_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect implementation for Concatenate."""
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


def normalized_concatenate_command(
    input_filenames: list[str],
    output_filename: str,
    data_colname: str,
) -> list[str]:
    """Return normalized command tokens for fixture comparisons."""
    return normalize_command(
        build_concatenate_command(input_filenames, output_filename, data_colname)
    )
