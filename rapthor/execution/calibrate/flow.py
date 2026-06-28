"""Prefect flows for the Calibrate operation."""

from typing import Mapping, Optional

from prefect import flow, task

import rapthor.execution.calibrate.runner as calibrate_runner
from rapthor.execution.calibrate.payloads import (
    CalibrateChunkPayload,
    CalibratePayload,
    validate_calibrate_payload,
)
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.task_runner import run_flow_with_task_runner


@task(name="calibrate_chunk")
def calibrate_chunk_task(
    payload: CalibratePayload,
    chunk: CalibrateChunkPayload,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one calibration chunk."""
    with publish_python_logs_to_prefect():
        return calibrate_runner.run_calibrate_chunk(
            payload,
            chunk,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )


@task(name="calibrate_screen_chunk")
def calibrate_screen_chunk_task(
    payload: CalibratePayload,
    chunk: CalibrateChunkPayload,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one screen-generation chunk."""
    with publish_python_logs_to_prefect():
        return calibrate_runner.run_calibrate_screen_chunk(
            payload,
            chunk,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )


def run_calibrate_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run calibration commands and return finalizer-compatible outputs."""
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    payload = validate_calibrate_payload(payload)
    payload = calibrate_runner.prepare_image_based_predict(
        payload,
        config,
        shell_operation_cls=shell_operation_cls,
    )
    if payload["calibration_kind"] == "dd_screen":
        screen_records = []
        for chunk in payload["chunks"]:
            screen_records.append(
                calibrate_runner.run_calibrate_screen_chunk(
                    payload,
                    chunk,
                    execution_config=config,
                    shell_operation_cls=shell_operation_cls,
                )
            )
        return calibrate_runner.collect_screen_solutions(
            payload,
            screen_records,
            config,
            shell_operation_cls=shell_operation_cls,
        )

    solve_records = []
    for chunk in payload["chunks"]:
        solve_records.append(
            calibrate_runner.run_calibrate_chunk(
                payload,
                chunk,
                execution_config=config,
                shell_operation_cls=shell_operation_cls,
            )
        )
    return calibrate_runner.collect_plot_and_combine(
        payload,
        solve_records,
        config,
        shell_operation_cls=shell_operation_cls,
    )


def _run_calibrate_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    config = execution_config or ExecutionConfig(task_runner="sync")
    payload = validate_calibrate_payload(payload)
    payload = calibrate_runner.prepare_image_based_predict(payload, config)
    if payload["calibration_kind"] == "dd_screen":
        screen_records = [
            calibrate_screen_chunk_task.submit(payload, chunk, execution_config=config)
            for chunk in payload["chunks"]
        ]
        screen_records = [record.result() for record in screen_records]
        return calibrate_runner.collect_screen_solutions(payload, screen_records, config)

    solve_records = [
        calibrate_chunk_task.submit(payload, chunk, execution_config=config)
        for chunk in payload["chunks"]
    ]
    solve_records = [record.result() for record in solve_records]
    return calibrate_runner.collect_plot_and_combine(payload, solve_records, config)


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
        execution_config=execution_config,
    )
