"""Prefect flows for imaging."""

from typing import Mapping, Optional

from prefect import flow, task
from prefect.exceptions import UnfinishedRun

import rapthor.execution.image.sector as image_sector
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.image.contracts import ImageSectorPayload
from rapthor.execution.image.validation import validate_image_payload
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.run_names import operation_run_name, task_run_name
from rapthor.execution.task_runner import run_flow_with_task_runner
from rapthor.lib.records import validate_output_record


@task(name="sector")
def image_sector_task(
    sector: ImageSectorPayload,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one imaging sector."""
    with publish_python_logs_to_prefect():
        return image_sector.run_image_sector(
            sector,
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )


@task(name="prepare")
def image_sector_prepare_task(
    sector: ImageSectorPayload,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for data preparation and WSClean for one sector."""
    with publish_python_logs_to_prefect():
        prepared = image_sector.prepare_image_sector(
            sector,
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(prepared)
    return prepared


@task(name="filter_skymodel")
def image_sector_filter_skymodel_task(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one sector skymodel-filtering step."""
    assert_serializable_payload(prepared)
    with publish_python_logs_to_prefect():
        filtered = image_sector.filter_image_sector_skymodel(
            sector,
            prepared,
            pipeline_working_dir,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(filtered)
    return filtered


@task(name="calculate_image_diagnostics")
def image_sector_diagnostics_task(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    filtered: Mapping[str, object],
    pipeline_working_dir: str,
) -> dict:
    """Prefect task wrapper for one sector diagnostics calculation."""
    assert_serializable_payload(prepared)
    assert_serializable_payload(filtered)
    with publish_python_logs_to_prefect():
        diagnostics = image_sector.calculate_image_sector_diagnostics(
            sector,
            prepared,
            filtered,
            pipeline_working_dir,
        )
    assert_serializable_payload(diagnostics)
    return diagnostics


@task(name="normalize_flux_scale")
def image_sector_normalize_flux_scale_task(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one sector flux-scale normalization step."""
    assert_serializable_payload(prepared)
    with publish_python_logs_to_prefect():
        normalization = image_sector.normalize_image_sector_flux_scale(
            sector,
            prepared,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    assert_serializable_payload(normalization)
    return normalization


@task(name="finalize")
def image_sector_finalize_task(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    filtered: Mapping[str, object],
    diagnostics: Mapping[str, object],
    pipeline_working_dir: str,
    normalization: Optional[Mapping[str, object]] = None,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for post-WSClean sector finalization."""
    assert_serializable_payload(prepared)
    assert_serializable_payload(filtered)
    assert_serializable_payload(diagnostics)
    assert_serializable_payload(normalization)
    with publish_python_logs_to_prefect():
        return image_sector.finalize_image_sector(
            sector,
            prepared,
            filtered,
            diagnostics,
            pipeline_working_dir,
            normalization_result=normalization,
            execution_config=execution_config,
            shell_operation_cls=shell_operation_cls,
        )


def _result_from_sector_records(sector_outputs: list[dict]) -> dict:
    result = {
        "filtered_skymodel_true_sky": [
            sector["filtered_skymodel_true_sky"] for sector in sector_outputs
        ],
        "filtered_skymodel_apparent_sky": [
            sector["filtered_skymodel_apparent_sky"] for sector in sector_outputs
        ],
        "pybdsf_catalog": [sector["pybdsf_catalog"] for sector in sector_outputs],
        "sector_diagnostics": [sector["sector_diagnostics"] for sector in sector_outputs],
        "sector_offsets": [sector["sector_offsets"] for sector in sector_outputs],
        "sector_diagnostic_plots": [sector["sector_diagnostic_plots"] for sector in sector_outputs],
        "visibilities": [sector["visibilities"] for sector in sector_outputs],
        "sector_I_images": [sector["sector_I_images"] for sector in sector_outputs],
        "sector_extra_images": [sector["sector_extra_images"] for sector in sector_outputs],
        "source_filtering_mask": [sector["source_filtering_mask"] for sector in sector_outputs],
    }
    if any(sector["sector_skymodels"] is not None for sector in sector_outputs):
        result["sector_skymodels"] = [sector["sector_skymodels"] for sector in sector_outputs]
    if any("sector_region_file" in sector for sector in sector_outputs):
        result["sector_region_file"] = [
            sector.get("sector_region_file") for sector in sector_outputs
        ]
    if any("sector_skymodel_image_fits" in sector for sector in sector_outputs):
        result["sector_skymodel_image_fits"] = [
            sector.get("sector_skymodel_image_fits") for sector in sector_outputs
        ]
    if any("sector_image_cubes" in sector for sector in sector_outputs):
        result["sector_image_cubes"] = [
            sector.get("sector_image_cubes") for sector in sector_outputs
        ]
        result["sector_image_cube_beams"] = [
            sector.get("sector_image_cube_beams") for sector in sector_outputs
        ]
        result["sector_image_cube_frequencies"] = [
            sector.get("sector_image_cube_frequencies") for sector in sector_outputs
        ]
    if any("sector_normalize_h5parm" in sector for sector in sector_outputs):
        result["sector_source_catalog"] = [
            sector.get("sector_source_catalog") for sector in sector_outputs
        ]
        result["sector_normalize_h5parm"] = [
            sector.get("sector_normalize_h5parm") for sector in sector_outputs
        ]
    if any("residual_visibilities" in sector for sector in sector_outputs):
        result["residual_visibilities"] = [
            sector.get("residual_visibilities") for sector in sector_outputs
        ]
    for value in result.values():
        validate_output_record(value, allow_none=True)
    return result


def _submit_split_image_sector_tasks(
    payload: Mapping[str, object],
    config: ExecutionConfig,
):
    sectors = list(payload["sectors"])

    def sector_step_name(index: int, step: str) -> str:
        if len(sectors) == 1:
            return task_run_name(step)
        sector = sectors[index]
        return task_run_name(sector.get("image_name") or f"sector_{index + 1}", step)

    prepared_sector_futures = [
        image_sector_prepare_task.with_options(
            task_run_name=sector_step_name(index, "prepare")
        ).submit(
            sector,
            payload["pipeline_working_dir"],
            execution_config=config,
        )
        for index, sector in enumerate(sectors)
    ]
    filtered_sector_futures = [
        image_sector_filter_skymodel_task.with_options(
            task_run_name=sector_step_name(index, "filter_skymodel")
        ).submit(
            sector,
            prepared_sector_futures[index],
            payload["pipeline_working_dir"],
            execution_config=config,
        )
        for index, sector in enumerate(sectors)
    ]
    diagnostics_sector_futures = [
        image_sector_diagnostics_task.with_options(
            task_run_name=sector_step_name(index, "calculate_image_diagnostics")
        ).submit(
            sector,
            prepared_sector_futures[index],
            filtered_sector_futures[index],
            payload["pipeline_working_dir"],
        )
        for index, sector in enumerate(sectors)
    ]
    normalization_sector_futures = [
        (
            image_sector_normalize_flux_scale_task.with_options(
                task_run_name=sector_step_name(index, "normalize_flux_scale")
            ).submit(
                sector,
                prepared_sector_futures[index],
                execution_config=config,
            )
            if bool(sector.get("normalize_flux_scale", False))
            else None
        )
        for index, sector in enumerate(sectors)
    ]
    finalized_sector_futures = [
        image_sector_finalize_task.with_options(
            task_run_name=sector_step_name(index, "finalize")
        ).submit(
            sector,
            prepared_sector_futures[index],
            filtered_sector_futures[index],
            diagnostics_sector_futures[index],
            payload["pipeline_working_dir"],
            normalization_sector_futures[index],
            execution_config=config,
        )
        for index, sector in enumerate(sectors)
    ]
    return (
        prepared_sector_futures,
        filtered_sector_futures,
        diagnostics_sector_futures,
        normalization_sector_futures,
        finalized_sector_futures,
    )


def _collect_image_sector_results(
    prepared_sector_futures,
    filtered_sector_futures,
    diagnostics_sector_futures,
    normalization_sector_futures,
    finalized_sector_futures,
) -> list[dict]:
    try:
        return [output.result() for output in finalized_sector_futures]
    except UnfinishedRun:
        for prepared_sector in prepared_sector_futures:
            prepared_sector.result()
        for filtered_sector in filtered_sector_futures:
            filtered_sector.result()
        for diagnostics_sector in diagnostics_sector_futures:
            diagnostics_sector.result()
        for normalization_sector in normalization_sector_futures:
            if normalization_sector is not None:
                normalization_sector.result()
        raise


def _run_image_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    payload = validate_image_payload(payload)
    (
        prepared_sector_futures,
        filtered_sector_futures,
        diagnostics_sector_futures,
        normalization_sector_futures,
        finalized_sector_futures,
    ) = _submit_split_image_sector_tasks(payload, config)
    sector_outputs = _collect_image_sector_results(
        prepared_sector_futures,
        filtered_sector_futures,
        diagnostics_sector_futures,
        normalization_sector_futures,
        finalized_sector_futures,
    )
    return _result_from_sector_records(sector_outputs)


@flow(name="image")
def _image_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect implementation for imaging."""
    with publish_python_logs_to_prefect():
        return _run_image_prefect_tasks(payload, execution_config=execution_config)


def image_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect entry point for imaging."""
    return run_flow_with_task_runner(
        _image_flow,
        payload,
        flow_run_name=operation_run_name(payload, "image"),
        execution_config=execution_config,
    )
