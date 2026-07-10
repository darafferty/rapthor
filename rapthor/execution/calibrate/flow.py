"""Prefect flows for the Calibrate operation."""

from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.calibrate.collection import (
    collect_screen_solutions,
    collect_strategy_solve_h5parm,
    process_plot_and_combine_collected_products,
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


@task(name="collect_h5parms")
def collect_h5parms_task(
    payload: CalibratePayload,
    solve_records: list[dict],
    solve_slot: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for collecting one solve slot's h5parm outputs."""
    assert_serializable_payload(solve_records)
    assert_serializable_payload(solve_slot)
    config = execution_config or ExecutionConfig(task_runner="sync")
    with publish_python_logs_to_prefect():
        collected = collect_strategy_solve_h5parm(
            payload,
            solve_records,
            solve_slot,
            config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(collected)
    return collected


@task(name="finalize_solutions")
def finalize_solutions_task(
    payload: CalibratePayload,
    collected_products: list[dict],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for calibration solution processing, plotting, and combining."""
    assert_serializable_payload(collected_products)
    config = execution_config or ExecutionConfig(task_runner="sync")
    with publish_python_logs_to_prefect():
        result = process_plot_and_combine_collected_products(
            payload,
            collected_products,
            config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(result)
    return result


@task(name="collect_screen_h5parms")
def collect_screen_h5parms_task(
    payload: CalibratePayload,
    screen_records: list[dict],
) -> dict:
    """Prefect task wrapper for collecting screen-generation h5parms."""
    assert_serializable_payload(screen_records)
    with publish_python_logs_to_prefect():
        result = collect_screen_solutions(payload, screen_records)
    assert_serializable_payload(result)
    return result


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
        return (
            collect_screen_h5parms_task.with_options(
                task_run_name=task_run_name("collect_screen_h5parms")
            )
            .submit(payload, screen_records)
            .result()
        )

    solve_records = [
        calibrate_chunk_task.with_options(task_run_name=task_run_name("chunk", index + 1)).submit(
            payload, chunk, execution_config=config
        )
        for index, chunk in enumerate(payload["chunks"])
    ]
    solve_records = [record.result() for record in solve_records]
    collected_products = [
        collect_h5parms_task.with_options(
            task_run_name=task_run_name("collect_h5parms", index + 1)
        ).submit(
            payload,
            solve_records,
            solve_slot,
            execution_config=config,
        )
        for index, solve_slot in enumerate(payload["chunks"][0]["solve_slots"])
    ]
    collected_products = [record.result() for record in collected_products]
    return (
        finalize_solutions_task.with_options(task_run_name=task_run_name("finalize_solutions"))
        .submit(payload, collected_products, execution_config=config)
        .result()
    )


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
