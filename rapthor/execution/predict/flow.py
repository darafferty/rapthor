"""Prefect flows for the Predict operation."""

import glob
import os
from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.outputs import require_directory
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.predict.commands import (
    build_predict_model_data_command,
)
from rapthor.execution.predict.payloads import (
    PredictModelTaskPayload,
    PredictPostprocessPayload,
    validate_predict_payload,
)
from rapthor.execution.predict.sector_model_addition import add_sector_models
from rapthor.execution.predict.sector_model_subtraction import subtract_sector_models
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.run_names import operation_run_name, task_run_options
from rapthor.execution.shell import run_external_command
from rapthor.execution.task_metrics import record_task_runtime
from rapthor.execution.task_runner import run_flow_with_task_runner
from rapthor.lib.records import directory_record, validate_output_record


def _run_command_and_validate_directory(
    command: list[str],
    output_path: str,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    run_external_command(
        command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    return require_directory(output_path, "Predict output")


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
    return _run_command_and_validate_directory(
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
) -> list[dict]:
    """Run DI add or DD subtract post-processing for one observation."""
    model_paths = [str(record["path"]) for record in model_outputs]
    if mode == "di":
        add_sector_models(
            postprocess_task["msobs"],
            model_paths,
            msin_column=postprocess_task["data_colname"],
            starttime=postprocess_task["obs_starttime"],
            infix=postprocess_task["infix"],
            output_dir=pipeline_working_dir,
        )
        output_patterns = [
            os.path.join(
                pipeline_working_dir,
                f"{os.path.basename(postprocess_task['msobs'])}*_di.ms",
            )
        ]
    elif mode == "dd":
        dd_task = postprocess_task
        subtract_sector_models(
            dd_task["msobs"],
            model_paths,
            msin_column=dd_task["data_colname"],
            starttime=dd_task["obs_starttime"],
            solint_sec=dd_task["solint_sec"],
            solint_hz=dd_task["solint_hz"],
            weights_colname="WEIGHT_SPECTRUM",
            uvcut_min=dd_task["min_uv_lambda"],
            uvcut_max=dd_task["max_uv_lambda"],
            phaseonly=True,
            nr_outliers=dd_task["nr_outliers"],
            peel_outliers=dd_task["peel_outliers"],
            nr_bright=dd_task["nr_bright"],
            peel_bright=dd_task["peel_bright"],
            reweight=dd_task["reweight"],
            infix=dd_task["infix"],
            output_dir=pipeline_working_dir,
        )
        obs_basename = os.path.basename(dd_task["msobs"])
        output_patterns = [
            os.path.join(pipeline_working_dir, f"{obs_basename}*_field"),
            os.path.join(pipeline_working_dir, f"{obs_basename}*.sector_*"),
        ]
    else:
        raise ValueError("mode must be 'di' or 'dd'")

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
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        return run_predict_model_data(
            predict_task,
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )


@task(name="postprocess")
def predict_postprocess_task(
    mode: str,
    postprocess_task: PredictPostprocessPayload,
    model_outputs: list[dict],
    pipeline_working_dir: str,
) -> list[dict]:
    """Prefect task wrapper for DI add or DD subtract post-processing."""
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        return run_predict_postprocess(
            mode,
            postprocess_task,
            model_outputs,
            pipeline_working_dir,
        )


def _result_from_postprocess_records(mode: str, outputs: list[list[dict]]) -> dict:
    key = "msfiles_di_cal" if mode == "di" else "subtract_models"
    result = {key: outputs}
    validate_output_record(result[key])
    return result


def _predict_task_matches_postprocess(
    predict_task: PredictModelTaskPayload,
    postprocess_task: PredictPostprocessPayload,
) -> bool:
    """Return True when a model-data task feeds one observation postprocess task."""
    model_output_name = os.path.basename(predict_task["msout"])
    observation_name = os.path.basename(postprocess_task["msobs"])
    same_msin = predict_task["msin"] == postprocess_task["msobs"]
    same_output_prefix = model_output_name.startswith(observation_name)
    same_starttime = predict_task["starttime"] == postprocess_task["obs_starttime"]
    return (same_msin or same_output_prefix) and same_starttime


def _model_output_futures_for_postprocess(
    postprocess_task: PredictPostprocessPayload,
    predict_tasks: list[PredictModelTaskPayload],
    model_output_futures: list[object],
) -> list[object]:
    """Select the model-data futures needed by one observation postprocess task."""
    matched_futures = [
        future
        for predict_task, future in zip(predict_tasks, model_output_futures)
        if _predict_task_matches_postprocess(predict_task, postprocess_task)
    ]
    if not matched_futures:
        raise ValueError(
            "No predict model-data task matches postprocess observation "
            f"{postprocess_task['msobs']} at start time {postprocess_task['obs_starttime']}"
        )
    return matched_futures


def _run_predict_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    payload = validate_predict_payload(payload)
    model_outputs = [
        predict_model_data_task.with_options(
            **task_run_options("predict_model_data", index + 1, tags=["dp3"])
        ).submit(
            predict_task,
            payload["pipeline_working_dir"],
            execution_config=config,
        )
        for index, predict_task in enumerate(payload["predict_tasks"])
    ]
    postprocess_outputs = [
        predict_postprocess_task.with_options(
            **task_run_options("postprocess", index + 1, tags=["python", "casacore"])
        ).submit(
            payload["mode"],
            postprocess_task,
            _model_output_futures_for_postprocess(
                postprocess_task,
                payload["predict_tasks"],
                model_outputs,
            ),
            payload["pipeline_working_dir"],
        )
        for index, postprocess_task in enumerate(payload["postprocess_tasks"])
    ]
    try:
        postprocess_outputs = [output.result() for output in postprocess_outputs]
    except Exception:
        for model_output in model_outputs:
            try:
                model_output.result()
            except Exception:
                raise
        raise
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
        flow_run_name=operation_run_name(payload, "predict", mode=payload.get("mode")),
        execution_config=execution_config,
    )
