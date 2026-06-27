"""Prefect flows for imaging."""

from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.image.payloads import ImageSectorPayload, validate_image_payload
from rapthor.execution.image.sector import run_image_sector as _run_image_sector
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.runtime import run_flow_with_task_runner
from rapthor.lib.records import validate_output_record


@task(name="image_sector")
def image_sector_task(
    sector: ImageSectorPayload,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one imaging sector."""
    with publish_python_logs_to_prefect():
        return _run_image_sector(
            sector,
            pipeline_working_dir,
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
    for value in result.values():
        validate_output_record(value, allow_none=True)
    return result


def run_image_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run imaging commands and return finalizer-compatible outputs."""
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    payload = validate_image_payload(payload)
    sector_outputs = [
        _run_image_sector(
            sector,
            payload["pipeline_working_dir"],
            execution_config=config,
            shell_operation_cls=shell_operation_cls,
        )
        for sector in payload["sectors"]
    ]
    return _result_from_sector_records(sector_outputs)


def _run_image_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    config = execution_config or ExecutionConfig(task_runner="sync")
    payload = validate_image_payload(payload)
    sector_outputs = [
        image_sector_task.submit(
            sector,
            payload["pipeline_working_dir"],
            execution_config=config,
        )
        for sector in payload["sectors"]
    ]
    sector_outputs = [output.result() for output in sector_outputs]
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
        execution_config=execution_config,
    )
