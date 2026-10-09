"""Predict execution payload builders and validators."""

import os
from typing import Mapping, Optional, TypedDict, Union

from rapthor.execution.payloads import (
    assert_serializable_payload,
    optional_string,
    validate_basename,
    validate_string_list,
)
from rapthor.lib.records import (
    directory_record_path,
    file_record_path,
    optional_file_record_path,
)


class PredictModelTaskPayload(TypedDict):
    """Serializable payload for one DP3 model prediction task."""

    msin: str
    data_colname: str
    msout: str
    msout_path: str
    starttime: str
    ntimes: int
    onebeamperpatch: bool
    correctfreqsmearing: bool
    correcttimesmearing: bool
    sagecalpredict: bool
    sourcedb: str
    directions: list[str]
    numthreads: int
    h5parm: Optional[str]
    applycal_steps: Optional[str]
    normalize_h5parm: Optional[str]


class PredictPostprocessTaskPayload(TypedDict):
    """Serializable payload for one DI predict post-processing task."""

    msobs: str
    data_colname: str
    obs_starttime: str
    infix: str


class PredictDDPostprocessTaskPayload(PredictPostprocessTaskPayload):
    """Serializable DD-specific extension for predict post-processing."""

    solint_sec: float
    solint_hz: float
    min_uv_lambda: float
    max_uv_lambda: float
    nr_outliers: int
    peel_outliers: bool
    nr_bright: int
    peel_bright: bool
    reweight: bool


PredictPostprocessPayload = Union[
    PredictPostprocessTaskPayload,
    PredictDDPostprocessTaskPayload,
]


class PredictPayload(TypedDict):
    """Serializable payload submitted to the predict flow."""

    mode: str
    pipeline_working_dir: str
    predict_tasks: list[PredictModelTaskPayload]
    postprocess_tasks: list[PredictPostprocessPayload]


def predict_payload_from_inputs(
    mode: str,
    input_parms: Mapping[str, object],
    pipeline_working_dir: object,
) -> PredictPayload:
    """Create a serializable Predict flow payload from operation inputs."""
    if mode not in {"di", "dd"}:
        raise ValueError("mode must be 'di' or 'dd'")
    pipeline_dir = str(pipeline_working_dir)

    sector_filenames = input_parms.get("sector_filename", [])
    sector_model_filenames = input_parms.get("sector_model_filename", [])
    sector_starttimes = input_parms.get("sector_starttime", [])
    sector_ntimes = input_parms.get("sector_ntimes", [])
    sector_skymodels = input_parms.get("sector_skymodel", [])
    sector_patches = input_parms.get("sector_patches", [])
    predict_inputs = [
        sector_filenames,
        sector_model_filenames,
        sector_starttimes,
        sector_ntimes,
        sector_skymodels,
        sector_patches,
    ]
    if not all(isinstance(value, list) for value in predict_inputs):
        raise ValueError("Predict scatter inputs must be lists")
    predict_count = len(sector_model_filenames)
    if any(len(value) != predict_count for value in predict_inputs):
        raise ValueError("Predict scatter inputs must have the same length")

    data_colname = str(input_parms["data_colname"])
    h5parm = optional_file_record_path(input_parms.get("h5parm"))
    normalize_h5parm = optional_file_record_path(input_parms.get("normalize_h5parm"))
    dp3_applycal_steps = optional_string(input_parms.get("dp3_applycal_steps"))
    predict_tasks: list[PredictModelTaskPayload] = []
    for index in range(predict_count):
        msout = validate_basename(sector_model_filenames[index], f"sector_model_filename[{index}]")
        if not isinstance(sector_patches[index], list):
            raise ValueError(f"sector_patches[{index}] must be a list")
        predict_tasks.append(
            {
                "msin": directory_record_path(sector_filenames[index]),
                "data_colname": data_colname,
                "msout": msout,
                "msout_path": os.path.join(pipeline_dir, msout),
                "starttime": str(sector_starttimes[index]),
                "ntimes": int(sector_ntimes[index]),
                "onebeamperpatch": bool(input_parms["onebeamperpatch"]),
                "correctfreqsmearing": bool(input_parms["correctfreqsmearing"]),
                "correcttimesmearing": bool(input_parms["correcttimesmearing"]),
                "sagecalpredict": bool(input_parms["sagecalpredict"]),
                "sourcedb": file_record_path(sector_skymodels[index]),
                "directions": [str(direction) for direction in sector_patches[index]],
                "numthreads": int(input_parms["max_threads"]),
                "h5parm": h5parm,
                "applycal_steps": dp3_applycal_steps,
                "normalize_h5parm": normalize_h5parm,
            }
        )

    obs_filenames = input_parms.get("obs_filename", [])
    obs_starttimes = input_parms.get("obs_starttime", [])
    obs_infixes = input_parms.get("obs_infix", [])
    obs_inputs = [obs_filenames, obs_starttimes, obs_infixes]
    if mode == "dd":
        obs_inputs.extend(
            [input_parms.get("obs_solint_sec", []), input_parms.get("obs_solint_hz", [])]
        )
    if not all(isinstance(value, list) for value in obs_inputs):
        raise ValueError("Predict observation inputs must be lists")
    obs_count = len(obs_filenames)
    if any(len(value) != obs_count for value in obs_inputs):
        raise ValueError("Predict observation inputs must have the same length")

    postprocess_tasks: list[PredictPostprocessPayload] = []
    for index in range(obs_count):
        task: PredictPostprocessPayload = {
            "msobs": directory_record_path(obs_filenames[index]),
            "data_colname": data_colname,
            "obs_starttime": str(obs_starttimes[index]),
            "infix": str(obs_infixes[index]),
        }
        if mode == "dd":
            task = {
                **task,
                "solint_sec": float(input_parms["obs_solint_sec"][index]),
                "solint_hz": float(input_parms["obs_solint_hz"][index]),
                "min_uv_lambda": float(input_parms["min_uv_lambda"]),
                "max_uv_lambda": float(input_parms["max_uv_lambda"]),
                "nr_outliers": int(input_parms["nr_outliers"]),
                "peel_outliers": bool(input_parms["peel_outliers"]),
                "nr_bright": int(input_parms["nr_bright"]),
                "peel_bright": bool(input_parms["peel_bright"]),
                "reweight": bool(input_parms["reweight"]),
            }
        postprocess_tasks.append(task)

    payload: PredictPayload = {
        "mode": mode,
        "pipeline_working_dir": pipeline_dir,
        "predict_tasks": predict_tasks,
        "postprocess_tasks": postprocess_tasks,
    }
    assert_serializable_payload(payload)
    return payload


def validate_predict_payload(payload: Mapping[str, object]) -> PredictPayload:
    """Validate a Predict payload received by a flow or worker."""
    mode = str(payload["mode"])
    if mode not in {"di", "dd"}:
        raise ValueError("mode must be 'di' or 'dd'")
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    raw_predict_tasks = payload.get("predict_tasks", [])
    raw_postprocess_tasks = payload.get("postprocess_tasks", [])
    if not isinstance(raw_predict_tasks, list):
        raise ValueError("predict_tasks must be a list")
    if not isinstance(raw_postprocess_tasks, list):
        raise ValueError("postprocess_tasks must be a list")
    predict_tasks = []
    for index, predict_task in enumerate(raw_predict_tasks):
        if not isinstance(predict_task, Mapping):
            raise ValueError(f"predict_tasks[{index}] must be a mapping")
        predict_tasks.append(_validate_predict_model_task(predict_task, index))
    postprocess_tasks = []
    for index, postprocess_task in enumerate(raw_postprocess_tasks):
        if not isinstance(postprocess_task, Mapping):
            raise ValueError(f"postprocess_tasks[{index}] must be a mapping")
        postprocess_tasks.append(_validate_predict_postprocess_task(mode, postprocess_task, index))
    return {
        "mode": mode,
        "pipeline_working_dir": pipeline_working_dir,
        "predict_tasks": predict_tasks,
        "postprocess_tasks": postprocess_tasks,
    }


def _validate_predict_model_task(
    predict_task: Mapping[str, object],
    index: int,
) -> PredictModelTaskPayload:
    """Validate one DP3 model-prediction task inside a predict payload."""
    return {
        "msin": str(predict_task["msin"]),
        "data_colname": str(predict_task["data_colname"]),
        "msout": validate_basename(predict_task["msout"], f"predict_tasks[{index}].msout"),
        "msout_path": str(predict_task["msout_path"]),
        "starttime": str(predict_task["starttime"]),
        "ntimes": int(predict_task["ntimes"]),
        "onebeamperpatch": bool(predict_task["onebeamperpatch"]),
        "correctfreqsmearing": bool(predict_task["correctfreqsmearing"]),
        "correcttimesmearing": bool(predict_task["correcttimesmearing"]),
        "sagecalpredict": bool(predict_task["sagecalpredict"]),
        "sourcedb": str(predict_task["sourcedb"]),
        "directions": validate_string_list(
            predict_task.get("directions"),
            f"predict_tasks[{index}].directions",
        ),
        "numthreads": int(predict_task["numthreads"]),
        "h5parm": optional_string(predict_task.get("h5parm")),
        "applycal_steps": optional_string(predict_task.get("applycal_steps")),
        "normalize_h5parm": optional_string(predict_task.get("normalize_h5parm")),
    }


def _validate_predict_postprocess_task(
    mode: str,
    postprocess_task: Mapping[str, object],
    index: int,
) -> PredictPostprocessPayload:
    """Validate one observation post-processing task for DI or DD prediction."""
    task: PredictPostprocessPayload = {
        "msobs": str(postprocess_task["msobs"]),
        "data_colname": str(postprocess_task["data_colname"]),
        "obs_starttime": str(postprocess_task["obs_starttime"]),
        "infix": str(postprocess_task["infix"]),
    }
    if mode == "dd":
        task = {
            **task,
            "solint_sec": float(postprocess_task["solint_sec"]),
            "solint_hz": float(postprocess_task["solint_hz"]),
            "min_uv_lambda": float(postprocess_task["min_uv_lambda"]),
            "max_uv_lambda": float(postprocess_task["max_uv_lambda"]),
            "nr_outliers": int(postprocess_task["nr_outliers"]),
            "peel_outliers": bool(postprocess_task["peel_outliers"]),
            "nr_bright": int(postprocess_task["nr_bright"]),
            "peel_bright": bool(postprocess_task["peel_bright"]),
            "reweight": bool(postprocess_task["reweight"]),
        }
    return task
