"""Prefect flow for the Concatenate operation."""

import os
from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.commands import normalize_command
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.runtime import run_flow_with_task_runner
from rapthor.execution.outputs import (
    directory_record,
    directory_record_path,
    validate_output_record,
)
from rapthor.execution.payloads import (
    ConcatenateEpochPayload,
    ConcatenatePayload,
    assert_serializable_payload,
)
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.shell import ShellCommand, run_shell_command


def _validate_output_filename(output_filename: object, index: int) -> str:
    if not isinstance(output_filename, str) or not output_filename:
        raise ValueError(f"output_filenames[{index}] must be a non-empty string")
    if os.path.isabs(output_filename) or os.path.basename(output_filename) != output_filename:
        raise ValueError(f"output_filenames[{index}] must be a basename")
    return output_filename


def _validate_input_filenames(input_filenames: object, index: int) -> list[str]:
    if (
        not isinstance(input_filenames, list)
        or not input_filenames
        or not all(
            isinstance(input_filename, str) and input_filename for input_filename in input_filenames
        )
    ):
        raise ValueError(f"epochs[{index}].input_filenames must be a non-empty list of strings")
    return list(input_filenames)


def _validate_concatenate_payload(payload: Mapping[str, object]) -> ConcatenatePayload:
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    data_colname = str(payload["data_colname"])
    raw_epochs = payload.get("epochs", [])
    if not isinstance(raw_epochs, list):
        raise ValueError("epochs must be a list")
    epochs: list[ConcatenateEpochPayload] = []
    for index, epoch in enumerate(raw_epochs):
        if not isinstance(epoch, Mapping):
            raise ValueError(f"epochs[{index}] must be a mapping")
        input_filenames = _validate_input_filenames(epoch.get("input_filenames"), index)
        output_filename = _validate_output_filename(epoch.get("output_filename"), index)
        expected_output_path = os.path.join(pipeline_working_dir, output_filename)
        if str(epoch.get("output_path")) != expected_output_path:
            raise ValueError(f"epochs[{index}].output_path must be {expected_output_path}")
        epochs.append(
            {
                "input_filenames": input_filenames,
                "output_filename": output_filename,
                "output_path": expected_output_path,
            }
        )
    _validate_unique_output_paths(epochs)
    return {
        "pipeline_working_dir": pipeline_working_dir,
        "data_colname": data_colname,
        "epochs": epochs,
    }


def _validate_unique_output_paths(epochs: list[ConcatenateEpochPayload]) -> None:
    output_paths = [epoch["output_path"] for epoch in epochs]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("epoch output paths must be unique")


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
) -> ConcatenatePayload:
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

    epochs: list[ConcatenateEpochPayload] = []
    for index, (epoch_inputs, output_filename) in enumerate(zip(input_filenames, output_filenames)):
        if not isinstance(epoch_inputs, list):
            raise ValueError(f"input_filenames[{index}] must be a list")
        output_filename = _validate_output_filename(output_filename, index)
        epochs.append(
            {
                "input_filenames": [directory_record_path(record) for record in epoch_inputs],
                "output_filename": output_filename,
                "output_path": os.path.join(pipeline_dir, output_filename),
            }
        )

    payload: ConcatenatePayload = {
        "pipeline_working_dir": pipeline_dir,
        "data_colname": data_colname,
        "epochs": epochs,
    }
    _validate_unique_output_paths(epochs)
    assert_serializable_payload(payload)
    return payload


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


def run_concatenate_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run concatenation commands and return finalizer-compatible outputs."""
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    payload = _validate_concatenate_payload(payload)
    outputs = []

    for epoch in payload["epochs"]:
        outputs.append(
            run_concatenate_epoch(
                epoch,
                payload["data_colname"],
                payload["pipeline_working_dir"],
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
    payload = _validate_concatenate_payload(payload)
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


def normalized_concatenate_command(
    input_filenames: list[str],
    output_filename: str,
    data_colname: str,
) -> list[str]:
    """Return normalized command tokens for fixture comparisons."""
    return normalize_command(
        build_concatenate_command(input_filenames, output_filename, data_colname)
    )
