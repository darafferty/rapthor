"""Prefect flows for the Calibrate operation."""

from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.calibrate.collection import (
    active_solution_product_index,
    collect_screen_solutions,
    collect_strategy_solve_h5parm,
    combine_processed_solution_products,
    finalize_processed_solution_products,
    needs_solution_combination,
    plot_processed_solve_product,
    process_collected_solve_product,
)
from rapthor.execution.calibrate.contracts import (
    CalibrateChunkPayload,
    CalibratePayload,
)
from rapthor.execution.calibrate.prediction import (
    adjust_prediction_normalization_h5parm,
    draw_predict_model_images,
    make_predict_region_file,
    prepare_wsclean_predict_chunk,
    wsclean_predict_facet_info,
)
from rapthor.execution.calibrate.solves import run_calibrate_chunk, run_calibrate_screen_chunk
from rapthor.execution.calibrate.validation import validate_calibrate_payload
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.run_names import operation_run_name, task_run_name, task_run_options
from rapthor.execution.task_metrics import record_task_runtime
from rapthor.execution.task_runner import run_flow_with_task_runner


@task(name="chunk")
def calibrate_chunk_task(
    payload: CalibratePayload,
    chunk: CalibrateChunkPayload,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one calibration chunk."""
    with publish_python_logs_to_prefect(), record_task_runtime(payload["pipeline_working_dir"]):
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
    with publish_python_logs_to_prefect(), record_task_runtime(payload["pipeline_working_dir"]):
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
    with publish_python_logs_to_prefect(), record_task_runtime(payload["pipeline_working_dir"]):
        collected = collect_strategy_solve_h5parm(
            payload,
            solve_records,
            solve_slot,
            config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(collected)
    return collected


@task(name="process_solutions")
def process_solutions_task(
    payload: CalibratePayload,
    collected_product: dict,
) -> dict:
    """Prefect task wrapper for per-solve calibration solution processing."""
    assert_serializable_payload(collected_product)
    with publish_python_logs_to_prefect(), record_task_runtime(payload["pipeline_working_dir"]):
        result = process_collected_solve_product(payload, collected_product)
    assert_serializable_payload(result)
    return result


@task(name="plot_solutions")
def plot_solutions_task(
    payload: CalibratePayload,
    processed_product: dict,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for per-solve calibration solution plots."""
    assert_serializable_payload(processed_product)
    config = execution_config or ExecutionConfig(task_runner="sync")
    with publish_python_logs_to_prefect(), record_task_runtime(payload["pipeline_working_dir"]):
        result = plot_processed_solve_product(
            payload,
            processed_product,
            config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(result)
    return result


@task(name="combine_h5parms")
def combine_h5parms_task(
    payload: CalibratePayload,
    processed_products: list[dict],
) -> dict:
    """Prefect task wrapper for combining processed calibration h5parms."""
    assert_serializable_payload(processed_products)
    with publish_python_logs_to_prefect(), record_task_runtime(payload["pipeline_working_dir"]):
        result = combine_processed_solution_products(payload, processed_products)
    assert_serializable_payload(result)
    return result


@task(name="finalize_solutions")
def finalize_solutions_task(
    payload: CalibratePayload,
    processed_products: list[dict],
    plot_products: list[dict],
    active_solution: dict,
) -> dict:
    """Prefect task wrapper for collecting finalized calibration output records."""
    assert_serializable_payload(processed_products)
    assert_serializable_payload(plot_products)
    assert_serializable_payload(active_solution)
    with publish_python_logs_to_prefect(), record_task_runtime(payload["pipeline_working_dir"]):
        result = finalize_processed_solution_products(
            payload,
            processed_products,
            plot_products,
            active_solution,
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
    with publish_python_logs_to_prefect(), record_task_runtime(payload["pipeline_working_dir"]):
        result = collect_screen_solutions(payload, screen_records)
    assert_serializable_payload(result)
    return result


@task(name="make_predict_region")
def make_predict_region_task(payload: CalibratePayload) -> dict:
    """Prefect task wrapper for preparing a calibration prediction region."""
    assert_serializable_payload(payload)
    with publish_python_logs_to_prefect(), record_task_runtime(payload["pipeline_working_dir"]):
        result = make_predict_region_file(payload)
    assert_serializable_payload(result)
    return result


@task(name="draw_model")
def draw_predict_model_task(
    payload: CalibratePayload,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> list[dict]:
    """Prefect task wrapper for drawing calibration prediction model images."""
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    with publish_python_logs_to_prefect(), record_task_runtime(payload["pipeline_working_dir"]):
        result = draw_predict_model_images(
            payload,
            config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(result)
    return result


@task(name="read_predict_facets")
def wsclean_predict_facet_info_task(
    region_record: Mapping[str, object],
    pipeline_working_dir: str,
) -> dict:
    """Prefect task wrapper for reading WSClean-predict facet names."""
    assert_serializable_payload(region_record)
    with publish_python_logs_to_prefect(), record_task_runtime(pipeline_working_dir):
        result = wsclean_predict_facet_info(region_record)
    assert_serializable_payload(result)
    return result


@task(name="wsclean_predict")
def wsclean_predict_chunk_task(
    payload: CalibratePayload,
    chunk: CalibrateChunkPayload,
    chunk_index: int,
    facet_info: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one WSClean-predict calibration chunk."""
    assert_serializable_payload(payload)
    assert_serializable_payload(chunk)
    assert_serializable_payload(facet_info)
    config = execution_config or ExecutionConfig(task_runner="sync")
    with publish_python_logs_to_prefect(), record_task_runtime(payload["pipeline_working_dir"]):
        result = prepare_wsclean_predict_chunk(
            payload,
            chunk,
            chunk_index,
            facet_info,
            config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(result)
    return result


@task(name="adjust_normalization_h5parm")
def adjust_prediction_normalization_h5parm_task(payload: CalibratePayload) -> str:
    """Prefect task wrapper for prediction-time normalization h5parm adjustment."""
    assert_serializable_payload(payload)
    with publish_python_logs_to_prefect(), record_task_runtime(payload["pipeline_working_dir"]):
        result = adjust_prediction_normalization_h5parm(payload)
    assert_serializable_payload(result)
    return result


def _solve_task_label(solve_slot: Mapping[str, object]) -> str:
    solve_type = str(solve_slot["solve_type"])
    if solve_type == "fast_phase":
        return "fast_phase"
    if solve_type == "medium_phase":
        return f"{solve_slot['solution_label']}_phase"
    if solve_type == "slow_gains":
        return "slow_gains"
    if solve_type == "full_jones":
        return "full_jones"
    return f"solve{solve_slot['slot']}"


def _solve_task_run_name(action: str, solve_slot: Mapping[str, object]) -> str:
    return task_run_name(action, _solve_task_label(solve_slot))


def _prepare_prediction_payload_with_tasks(
    payload: CalibratePayload,
    config: ExecutionConfig,
) -> CalibratePayload:
    """Prepare image-based calibration prediction inputs with visible tasks."""
    if not (payload.get("image_based_predict") or payload.get("wsclean_predict")):
        return payload

    region_future = make_predict_region_task.with_options(
        **task_run_options("make_predict_region", tags=["python"])
    ).submit(payload)

    model_images_future = None
    if payload.get("image_based_predict"):
        model_images_future = draw_predict_model_task.with_options(
            **task_run_options("draw_model", tags=["wsclean"])
        ).submit(payload, execution_config=config)

    facet_info_future = None
    prepared_chunk_futures = None
    if payload.get("wsclean_predict"):
        facet_info_future = wsclean_predict_facet_info_task.with_options(
            **task_run_options("read_predict_facets", tags=["python"])
        ).submit(region_future, payload["pipeline_working_dir"])
        prepared_chunk_futures = [
            wsclean_predict_chunk_task.with_options(
                **task_run_options("wsclean_predict", index + 1, tags=["wsclean"])
            ).submit(
                payload,
                chunk,
                index,
                facet_info_future,
                execution_config=config,
            )
            for index, chunk in enumerate(payload["chunks"])
        ]

    normalization_future = None
    if payload.get("normalize_h5parm"):
        normalization_future = adjust_prediction_normalization_h5parm_task.with_options(
            **task_run_options("adjust_normalization_h5parm", tags=["python"])
        ).submit(payload)

    prepared_payload = dict(payload)
    region_record = region_future.result()
    prepared_payload["predict_regions"] = region_record["path"]

    if model_images_future is not None:
        model_images = model_images_future.result()
        prepared_payload["predict_images"] = [record["path"] for record in model_images]

    if facet_info_future is not None and prepared_chunk_futures is not None:
        facet_info = facet_info_future.result()
        prepared_payload["modeldatacolumn"] = facet_info["modeldatacolumn"]
        prepared_payload["chunks"] = [future.result() for future in prepared_chunk_futures]

    if normalization_future is not None:
        prepared_payload["normalize_h5parm"] = normalization_future.result()

    assert_serializable_payload(prepared_payload)
    return prepared_payload


def _run_calibrate_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    payload = validate_calibrate_payload(payload)
    payload = _prepare_prediction_payload_with_tasks(payload, config)
    if payload["calibration_kind"] == "dd_screen":
        screen_records = [
            calibrate_screen_chunk_task.with_options(
                **task_run_options("screen_chunk", index + 1, tags=["dp3"])
            ).submit(payload, chunk, execution_config=config)
            for index, chunk in enumerate(payload["chunks"])
        ]
        screen_records = [record.result() for record in screen_records]
        return (
            collect_screen_h5parms_task.with_options(
                **task_run_options("collect_screen_h5parms", tags=["python"])
            )
            .submit(payload, screen_records)
            .result()
        )

    solve_records = [
        calibrate_chunk_task.with_options(
            **task_run_options("solve_chunk", index + 1, tags=["dp3"])
        ).submit(payload, chunk, execution_config=config)
        for index, chunk in enumerate(payload["chunks"])
    ]
    solve_records = [record.result() for record in solve_records]
    collected_products = [
        collect_h5parms_task.with_options(
            **task_run_options(_solve_task_run_name("collect", solve_slot), tags=["python"])
        ).submit(
            payload,
            solve_records,
            solve_slot,
            execution_config=config,
        )
        for solve_slot in payload["chunks"][0]["solve_slots"]
    ]
    solve_slots = list(payload["chunks"][0]["solve_slots"])
    processed_products = [
        process_solutions_task.with_options(
            **task_run_options(_solve_task_run_name("process", solve_slot), tags=["python"])
        ).submit(payload, collected_product)
        for collected_product, solve_slot in zip(collected_products, solve_slots)
    ]
    plot_products = [
        plot_solutions_task.with_options(
            **task_run_options(_solve_task_run_name("plot", solve_slot), tags=["python"])
        ).submit(payload, processed_product, execution_config=config)
        for processed_product, solve_slot in zip(processed_products, solve_slots)
    ]
    if needs_solution_combination(payload):
        active_solution = combine_h5parms_task.with_options(
            **task_run_options("combine_h5parms", tags=["python"])
        ).submit(payload, processed_products)
    else:
        active_solution = processed_products[active_solution_product_index(payload)]
    return (
        finalize_solutions_task.with_options(
            **task_run_options("finalize_solutions", tags=["python"])
        )
        .submit(payload, processed_products, plot_products, active_solution)
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
