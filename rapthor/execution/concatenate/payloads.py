"""Concatenate execution payload builders and validators."""

import os
from typing import Mapping, TypedDict

from rapthor.execution.payloads import (
    assert_serializable_payload,
    validate_basename,
    validate_string_list,
)
from rapthor.lib.records import directory_record_path


class ConcatenateEpochPayload(TypedDict):
    """Serializable inputs and expected output for one concatenate epoch."""

    input_filenames: list[str]
    output_filename: str
    output_path: str


class ConcatenatePayload(TypedDict):
    """Serializable payload submitted to the concatenate flow."""

    pipeline_working_dir: str
    data_colname: str
    epochs: list[ConcatenateEpochPayload]


def _validate_input_filenames(input_filenames: object, index: int) -> list[str]:
    return validate_string_list(
        input_filenames,
        f"epochs[{index}].input_filenames",
        allow_empty=False,
    )


def _validate_unique_output_paths(epochs: list[ConcatenateEpochPayload]) -> None:
    output_paths = [epoch["output_path"] for epoch in epochs]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("epoch output paths must be unique")


def validate_concatenate_payload(payload: Mapping[str, object]) -> ConcatenatePayload:
    """Validate a Concatenate payload received by a flow or worker."""
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
        output_filename = validate_basename(
            epoch.get("output_filename"), f"epochs[{index}].output_filename"
        )
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
        output_filename = validate_basename(output_filename, f"output_filenames[{index}]")
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
