"""Payload validators for calibration execution."""

from typing import Mapping, Optional

from rapthor.execution.calibrate.contracts import (
    CalibrateChunkPayload,
    CalibrateImagePredictPayload,
    CalibratePayload,
    CalibrateSolveSlotPayload,
)
from rapthor.execution.payloads import (
    validate_basename as _validate_basename,
)
from rapthor.execution.payloads import (
    validate_int_list as _validate_int_list,
)
from rapthor.execution.payloads import (
    validate_string_list as _validate_string_list,
)
from rapthor.lib.records import file_record_path


def _optional_file_path(record: object, name: str) -> Optional[str]:
    if record is None:
        return None
    if isinstance(record, str):
        return record
    if isinstance(record, Mapping) and record.get("class") == "File":
        return file_record_path(record)
    raise ValueError(f"{name} must be a File record, path string, or None")


def _validate_calibrate_solve_slot(
    solve_slot: Mapping[str, object],
    chunk_index: int,
    slot_index: int,
) -> CalibrateSolveSlotPayload:
    _ = int(solve_slot["slot"])
    _ = str(solve_slot["solve_type"])
    _ = str(solve_slot["solution_label"])
    if solve_slot.get("medium_index") is not None:
        _ = int(solve_slot["medium_index"])
    _validate_basename(
        solve_slot["h5parm"],
        f"chunks[{chunk_index}].solve_slots[{slot_index}].h5parm",
    )
    _ = str(solve_slot["h5parm_path"])
    _ = int(solve_slot["solint"])
    _ = str(solve_slot["mode"])
    _ = int(solve_slot["nchan"])
    for list_key in ("solutions_per_direction", "smoothness_dd_factors"):
        value = solve_slot.get(list_key)
        if value is not None and not isinstance(value, list):
            raise ValueError(
                f"chunks[{chunk_index}].solve_slots[{slot_index}].{list_key} must be a list"
            )
    return solve_slot


def _validate_calibrate_chunk(
    chunk: Mapping[str, object],
    index: int,
    *,
    screen: bool,
) -> CalibrateChunkPayload:
    _ = str(chunk["msin"])
    _ = str(chunk["starttime"])
    _ = int(chunk["ntimes"])
    _validate_basename(chunk["output_h5parm"], f"chunks[{index}].output_h5parm")
    _ = str(chunk["output_h5parm_path"])
    if screen:
        _ = int(chunk["solint_fast"])
        if "solint_slow" in chunk:
            _ = int(chunk["solint_slow"])
        return chunk

    raw_solve_slots = chunk.get("solve_slots", [])
    if not isinstance(raw_solve_slots, list) or not raw_solve_slots:
        raise ValueError(f"chunks[{index}].solve_slots must be a non-empty list")
    for slot_index, solve_slot in enumerate(raw_solve_slots):
        if not isinstance(solve_slot, Mapping):
            raise ValueError(f"chunks[{index}].solve_slots[{slot_index}] must be a mapping")
        _validate_calibrate_solve_slot(solve_slot, index, slot_index)
    return chunk


def _validate_calibrate_image_predict(
    image_predict: object,
) -> Optional[CalibrateImagePredictPayload]:
    if image_predict is None:
        return None
    if not isinstance(image_predict, Mapping):
        raise ValueError("image_predict must be a mapping")
    _ = _optional_file_path(image_predict.get("skymodel"), "image_predict.skymodel")
    _validate_basename(image_predict["model_image_root"], "image_predict.model_image_root")
    _validate_string_list(
        image_predict.get("model_image_ra_dec"), "image_predict.model_image_ra_dec"
    )
    _validate_int_list(
        image_predict.get("model_image_imsize"), "image_predict.model_image_imsize", length=2
    )
    _validate_string_list(image_predict.get("model_images"), "image_predict.model_images")
    _validate_basename(image_predict["facet_region_file"], "image_predict.facet_region_file")
    _ = str(image_predict["facet_region_path"])
    return image_predict


def validate_calibrate_payload(payload: Mapping[str, object]) -> CalibratePayload:
    """Validate an incoming Calibrate flow payload and return its typed shape."""
    mode = str(payload["mode"])
    if mode not in {"di", "dd"}:
        raise ValueError("mode must be 'di' or 'dd'")
    calibration_kind = str(payload["calibration_kind"])
    supported_kinds = {
        "di_fast_phase",
        "di_ddecal",
        "di_fulljones",
        "di_phase_slow",
        "di_scalar_phase",
        "di_slow",
        "dd_fast_phase",
        "dd_ddecal",
        "dd_phase",
        "dd_phase_slow",
        "dd_screen",
        "dd_slow",
    }
    if calibration_kind not in supported_kinds:
        raise ValueError("Unsupported calibration kind")
    _ = str(payload["pipeline_working_dir"])
    raw_chunks = payload.get("chunks", [])
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("chunks must be a non-empty list")
    for index, chunk in enumerate(raw_chunks):
        if not isinstance(chunk, Mapping):
            raise ValueError(f"chunks[{index}] must be a mapping")
        _validate_calibrate_chunk(chunk, index, screen=calibration_kind == "dd_screen")
    if payload.get("image_based_predict"):
        _validate_calibrate_image_predict(payload.get("image_predict"))
    return payload
