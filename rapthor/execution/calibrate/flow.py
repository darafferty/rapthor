"""Prefect flows for the Calibrate operation."""

from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.calibrate.collection import (
    collect_plot_and_combine,
    collect_screen_solutions,
)
from rapthor.execution.calibrate.contracts import (
    CalibrateChunkPayload,
    CalibratePayload,
)
from rapthor.execution.calibrate.prediction import prepare_image_based_predict
from rapthor.execution.calibrate.solves import run_calibrate_chunk, run_calibrate_screen_chunk
from rapthor.execution.calibrate.validation import validate_calibrate_payload
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.run_names import operation_run_name, task_run_name
from rapthor.execution.task_runner import run_flow_with_task_runner


@task(name="chunk")
def calibrate_chunk_task(
    payload: CalibratePayload,
    chunk: CalibrateChunkPayload,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one calibration chunk."""
    with publish_python_logs_to_prefect():
        return run_calibrate_chunk(
            payload,
            chunk,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )


@task(name="screen_chunk")
def calibrate_screen_chunk_task(
    payload: CalibratePayload,
    chunk: CalibrateChunkPayload,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one screen-generation chunk."""
    with publish_python_logs_to_prefect():
        return run_calibrate_screen_chunk(
            payload,
            chunk,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )


def _run_calibrate_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    payload = validate_calibrate_payload(payload)
    payload = prepare_image_based_predict(payload, config)
    if payload["calibration_kind"] == "dd_screen":
        screen_records = [
            calibrate_screen_chunk_task.with_options(
                task_run_name=task_run_name("screen", index + 1)
            ).submit(payload, chunk, execution_config=config)
            for index, chunk in enumerate(payload["chunks"])
        ]
        screen_records = [record.result() for record in screen_records]
        return collect_screen_solutions(payload, screen_records)

    solve_records = [
        calibrate_chunk_task.with_options(task_run_name=task_run_name("chunk", index + 1)).submit(
            payload, chunk, execution_config=config
        )
        for index, chunk in enumerate(payload["chunks"])
    ]
    solve_records = [record.result() for record in solve_records]
    return collect_plot_and_combine(payload, solve_records, config)


@flow(name="calibrate")
def _calibrate_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect implementation for Calibrate."""
    with publish_python_logs_to_prefect():
        return _run_calibrate_prefect_tasks(
            payload,
            execution_config=execution_config,
        )


def calibrate_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect entry point for Calibrate."""
    return run_flow_with_task_runner(
        _calibrate_flow,
        payload,
        flow_run_name=operation_run_name(payload, "calibrate", mode=payload.get("mode")),
        execution_config=execution_config,
    )
