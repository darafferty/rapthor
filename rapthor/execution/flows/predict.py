"""Prefect flows for the Predict operation."""

import glob
import os
from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.commands import normalize_command
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.runtime import run_flow_with_task_runner
from rapthor.execution.outputs import (
    directory_record,
    directory_record_path,
    file_record_path,
    optional_file_record_path,
    validate_output_record,
)
from rapthor.execution.payloads import (
    PredictModelTaskPayload,
    PredictPayload,
    PredictPostprocessPayload,
    assert_serializable_payload,
)
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.shell import ShellCommand, run_shell_command


def _bool_token(value: bool) -> str:
    return "True" if value else "False"


def _optional_str(value: object) -> Optional[str]:
    if value in (None, "", "None"):
        return None
    return str(value)


def _validate_basename(filename: object, name: str) -> str:
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{name} must be a non-empty string")
    if os.path.isabs(filename) or os.path.basename(filename) != filename:
        raise ValueError(f"{name} must be a basename")
    return filename


def _validate_string_list(values: object, name: str) -> list[str]:
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise ValueError(f"{name} must be a list of strings")
    return list(values)


def _predict_type(sagecalpredict: bool, h5parm: Optional[str]) -> str:
    if sagecalpredict:
        return "sagecalpredict"
    if h5parm is None:
        return "predict"
    return "h5parmpredict"


def _directions_token(directions: list[str]) -> str:
    return f"[{','.join(directions)}]"


def _join_path_list(paths: list[str]) -> str:
    return ",".join(paths)


def build_predict_model_data_command(
    msin: str,
    data_colname: str,
    msout: str,
    starttime: str,
    ntimes: int,
    onebeamperpatch: bool,
    correctfreqsmearing: bool,
    correcttimesmearing: bool,
    sagecalpredict: bool,
    sourcedb: str,
    directions: list[str],
    numthreads: int,
    h5parm: Optional[str] = None,
    applycal_steps: Optional[str] = None,
    normalize_h5parm: Optional[str] = None,
) -> list[str]:
    """Build the DP3 prediction command for one sector/observation pair."""
    command = [
        "DP3",
        "msout.overwrite=True",
        "steps=[predict]",
        "predict.operation=replace",
    ]
    if h5parm is not None:
        command.extend(
            [
                "predict.applycal.correction=phase000",
                "predict.applycal.fastphase.correction=phase000",
                "predict.applycal.fastphase.solset=sol000",
                "predict.applycal.slowgain.correction=amplitude000",
                "predict.applycal.slowgain.solset=sol000",
                "predict.applycal.normalization.correction=amplitude000",
                "predict.applycal.normalization.solset=sol000",
            ]
        )
    command.extend(
        [
            "predict.usebeammodel=True",
            "predict.beam_interval=120",
            "predict.beammode=array_factor",
            "msout.storagemanager=Dysco",
            "msout.storagemanager.databitrate=0",
            "msout.antennacompression=False",
            f"msin={msin}",
            f"msin.datacolumn={data_colname}",
            f"msout={msout}",
            f"msin.starttime={starttime}",
        ]
    )
    if ntimes > 0:
        command.append(f"msin.ntimes={ntimes}")
    command.extend(
        [
            f"predict.onebeamperpatch={_bool_token(onebeamperpatch)}",
            f"predict.correctfreqsmearing={_bool_token(correctfreqsmearing)}",
            f"predict.correcttimesmearing={_bool_token(correcttimesmearing)}",
            f"predict.type={_predict_type(sagecalpredict, h5parm)}",
        ]
    )
    if applycal_steps is not None:
        command.append(f"predict.applycal.steps={applycal_steps}")
    if h5parm is not None:
        command.append(f"predict.applycal.parmdb={h5parm}")
    if normalize_h5parm is not None:
        command.append(f"predict.applycal.normalization.parmdb={normalize_h5parm}")
    command.extend(
        [
            f"predict.sourcedb={sourcedb}",
            f"predict.directions={_directions_token(directions)}",
            f"numthreads={numthreads}",
        ]
    )
    return command


def build_add_sector_models_command(
    msobs: str,
    msmods: list[str],
    data_colname: str,
    obs_starttime: str,
    infix: str,
) -> list[str]:
    """Build the `add_sector_models.py` command for one observation."""
    return [
        "add_sector_models.py",
        msobs,
        _join_path_list(msmods),
        f"--msin_column={data_colname}",
        f"--starttime={obs_starttime}",
        f"--infix={infix}",
    ]


def build_subtract_sector_models_command(
    msobs: str,
    msmods: list[str],
    data_colname: str,
    obs_starttime: str,
    solint_sec: float,
    solint_hz: float,
    infix: str,
    min_uv_lambda: float,
    max_uv_lambda: float,
    nr_outliers: int,
    peel_outliers: bool,
    nr_bright: int,
    peel_bright: bool,
    reweight: bool,
) -> list[str]:
    """Build the `subtract_sector_models.py` command for one observation."""
    return [
        "subtract_sector_models.py",
        "--weights_colname=WEIGHT_SPECTRUM",
        "--phaseonly=True",
        msobs,
        _join_path_list(msmods),
        f"--msin_column={data_colname}",
        f"--starttime={obs_starttime}",
        f"--solint_sec={solint_sec}",
        f"--solint_hz={solint_hz}",
        f"--infix={infix}",
        f"--uvcut_min={min_uv_lambda}",
        f"--uvcut_max={max_uv_lambda}",
        f"--nr_outliers={nr_outliers}",
        f"--peel_outliers={_bool_token(peel_outliers)}",
        f"--nr_bright={nr_bright}",
        f"--peel_bright={_bool_token(peel_bright)}",
        f"--reweight={_bool_token(reweight)}",
    ]


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
    dp3_applycal_steps = _optional_str(input_parms.get("dp3_applycal_steps"))
    predict_tasks: list[PredictModelTaskPayload] = []
    for index in range(predict_count):
        msout = _validate_basename(sector_model_filenames[index], f"sector_model_filename[{index}]")
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


def _validate_predict_model_task(
    predict_task: Mapping[str, object],
    index: int,
) -> PredictModelTaskPayload:
    return {
        "msin": str(predict_task["msin"]),
        "data_colname": str(predict_task["data_colname"]),
        "msout": _validate_basename(predict_task["msout"], f"predict_tasks[{index}].msout"),
        "msout_path": str(predict_task["msout_path"]),
        "starttime": str(predict_task["starttime"]),
        "ntimes": int(predict_task["ntimes"]),
        "onebeamperpatch": bool(predict_task["onebeamperpatch"]),
        "correctfreqsmearing": bool(predict_task["correctfreqsmearing"]),
        "correcttimesmearing": bool(predict_task["correcttimesmearing"]),
        "sagecalpredict": bool(predict_task["sagecalpredict"]),
        "sourcedb": str(predict_task["sourcedb"]),
        "directions": _validate_string_list(
            predict_task.get("directions"),
            f"predict_tasks[{index}].directions",
        ),
        "numthreads": int(predict_task["numthreads"]),
        "h5parm": _optional_str(predict_task.get("h5parm")),
        "applycal_steps": _optional_str(predict_task.get("applycal_steps")),
        "normalize_h5parm": _optional_str(predict_task.get("normalize_h5parm")),
    }


def _validate_predict_postprocess_task(
    mode: str,
    postprocess_task: Mapping[str, object],
    index: int,
) -> PredictPostprocessPayload:
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


def _validate_predict_payload(payload: Mapping[str, object]) -> PredictPayload:
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


def _run_shell_and_validate_directory(
    command: list[str],
    output_path: str,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    run_shell_command(
        ShellCommand(command=command, working_directory=pipeline_working_dir),
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    if not os.path.isdir(output_path):
        raise FileNotFoundError(f"Predict output was not created: {output_path}")
    return directory_record(output_path)


def _glob_directory_records(
    patterns: list[str],
    exclude_suffixes: tuple[str, ...] = (),
) -> list[dict]:
    records = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isdir(path) and not path.endswith(exclude_suffixes):
                records.append(directory_record(path))
    if not records:
        patterns_text = ", ".join(patterns)
        raise FileNotFoundError(
            f"Predict post-processing outputs were not created: {patterns_text}"
        )
    return records


def run_predict_model_data(
    predict_task: PredictModelTaskPayload,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run one DP3 prediction task and return a directory record."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    command = build_predict_model_data_command(
        predict_task["msin"],
        predict_task["data_colname"],
        predict_task["msout"],
        predict_task["starttime"],
        predict_task["ntimes"],
        predict_task["onebeamperpatch"],
        predict_task["correctfreqsmearing"],
        predict_task["correcttimesmearing"],
        predict_task["sagecalpredict"],
        predict_task["sourcedb"],
        predict_task["directions"],
        predict_task["numthreads"],
        h5parm=predict_task["h5parm"],
        applycal_steps=predict_task["applycal_steps"],
        normalize_h5parm=predict_task["normalize_h5parm"],
    )
    return _run_shell_and_validate_directory(
        command,
        predict_task["msout_path"],
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )


def run_predict_postprocess(
    mode: str,
    postprocess_task: PredictPostprocessPayload,
    model_outputs: list[dict],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> list[dict]:
    """Run DI add or DD subtract post-processing for one observation."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    model_paths = [str(record["path"]) for record in model_outputs]
    if mode == "di":
        command = build_add_sector_models_command(
            postprocess_task["msobs"],
            model_paths,
            postprocess_task["data_colname"],
            postprocess_task["obs_starttime"],
            postprocess_task["infix"],
        )
        output_patterns = [
            os.path.join(
                pipeline_working_dir,
                f"{os.path.basename(postprocess_task['msobs'])}*_di.ms",
            )
        ]
    elif mode == "dd":
        dd_task = postprocess_task
        command = build_subtract_sector_models_command(
            dd_task["msobs"],
            model_paths,
            dd_task["data_colname"],
            dd_task["obs_starttime"],
            dd_task["solint_sec"],
            dd_task["solint_hz"],
            dd_task["infix"],
            dd_task["min_uv_lambda"],
            dd_task["max_uv_lambda"],
            dd_task["nr_outliers"],
            dd_task["peel_outliers"],
            dd_task["nr_bright"],
            dd_task["peel_bright"],
            dd_task["reweight"],
        )
        obs_basename = os.path.basename(dd_task["msobs"])
        output_patterns = [
            os.path.join(pipeline_working_dir, f"{obs_basename}*_field"),
            os.path.join(pipeline_working_dir, f"{obs_basename}*.sector_*"),
        ]
    else:
        raise ValueError("mode must be 'di' or 'dd'")

    run_shell_command(
        ShellCommand(command=command, working_directory=pipeline_working_dir),
        config,
        shell_operation_cls=shell_operation_cls,
    )
    exclude_suffixes = ("_modeldata",) if mode == "dd" else ()
    return _glob_directory_records(output_patterns, exclude_suffixes=exclude_suffixes)


@task(name="predict_model_data")
def predict_model_data_task(
    predict_task: PredictModelTaskPayload,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one DP3 prediction."""
    with publish_python_logs_to_prefect():
        return run_predict_model_data(
            predict_task,
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )


@task(name="predict_postprocess")
def predict_postprocess_task(
    mode: str,
    postprocess_task: PredictPostprocessPayload,
    model_outputs: list[dict],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> list[dict]:
    """Prefect task wrapper for DI add or DD subtract post-processing."""
    with publish_python_logs_to_prefect():
        return run_predict_postprocess(
            mode,
            postprocess_task,
            model_outputs,
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )


def _result_from_postprocess_records(mode: str, outputs: list[list[dict]]) -> dict:
    key = "msfiles_di_cal" if mode == "di" else "subtract_models"
    result = {key: outputs}
    validate_output_record(result[key])
    return result


def run_predict_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run Predict commands and return finalizer-compatible outputs."""
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    payload = _validate_predict_payload(payload)
    model_outputs = [
        run_predict_model_data(
            predict_task,
            payload["pipeline_working_dir"],
            execution_config=config,
            shell_operation_cls=shell_operation_cls,
        )
        for predict_task in payload["predict_tasks"]
    ]
    postprocess_outputs = [
        run_predict_postprocess(
            payload["mode"],
            postprocess_task,
            model_outputs,
            payload["pipeline_working_dir"],
            execution_config=config,
            shell_operation_cls=shell_operation_cls,
        )
        for postprocess_task in payload["postprocess_tasks"]
    ]
    return _result_from_postprocess_records(payload["mode"], postprocess_outputs)


def _run_predict_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    config = execution_config or ExecutionConfig(task_runner="sync")
    payload = _validate_predict_payload(payload)
    model_outputs = [
        predict_model_data_task.submit(
            predict_task,
            payload["pipeline_working_dir"],
            execution_config=config,
        )
        for predict_task in payload["predict_tasks"]
    ]
    model_outputs = [output.result() for output in model_outputs]
    postprocess_outputs = [
        predict_postprocess_task.submit(
            payload["mode"],
            postprocess_task,
            model_outputs,
            payload["pipeline_working_dir"],
            execution_config=config,
        )
        for postprocess_task in payload["postprocess_tasks"]
    ]
    postprocess_outputs = [output.result() for output in postprocess_outputs]
    return _result_from_postprocess_records(payload["mode"], postprocess_outputs)


@flow(name="predict")
def _predict_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect implementation for Predict."""
    with publish_python_logs_to_prefect():
        return _run_predict_prefect_tasks(payload, execution_config=execution_config)


def predict_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect entry point for Predict."""
    return run_flow_with_task_runner(
        _predict_flow,
        payload,
        execution_config=execution_config,
    )


def normalized_predict_model_data_command(**kwargs) -> list[str]:
    """Return normalized DP3 predict command tokens for fixture comparisons."""
    return normalize_command(build_predict_model_data_command(**kwargs))


def normalized_add_sector_models_command(**kwargs) -> list[str]:
    """Return normalized add-sector command tokens for fixture comparisons."""
    return normalize_command(build_add_sector_models_command(**kwargs))


def normalized_subtract_sector_models_command(**kwargs) -> list[str]:
    """Return normalized subtract-sector command tokens for fixture comparisons."""
    return normalize_command(build_subtract_sector_models_command(**kwargs))
