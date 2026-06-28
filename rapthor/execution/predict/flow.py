"""Prefect flows for the Predict operation."""

import glob
import os
from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.predict.commands import (
    build_add_sector_models_command,
    build_predict_model_data_command,
    build_subtract_sector_models_command,
)
from rapthor.execution.predict.payloads import (
    PredictModelTaskPayload,
    PredictPostprocessPayload,
    validate_predict_payload,
)
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.shell import ShellCommand, run_shell_command
from rapthor.execution.task_runner import run_flow_with_task_runner
from rapthor.lib.records import directory_record, validate_output_record


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
    payload = validate_predict_payload(payload)
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
    payload = validate_predict_payload(payload)
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
