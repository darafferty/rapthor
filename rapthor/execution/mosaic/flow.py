"""Prefect flow for the Mosaic operation."""

import os
from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.artifacts import publish_fits_image_artifacts
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.mosaic.commands import build_compress_mosaic_command
from rapthor.execution.mosaic.images import make_mosaic, make_mosaic_template, regrid_image
from rapthor.execution.mosaic.payloads import MosaicImageTypePayload, validate_mosaic_payload
from rapthor.execution.outputs import require_file
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.run_names import operation_run_name, task_run_name
from rapthor.execution.shell import run_external_command
from rapthor.execution.task_runner import run_flow_with_task_runner
from rapthor.lib.records import validate_output_record


def _run_command_and_require_file(
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
    return require_file(output_path, "Mosaic output")


def _work_path(pipeline_working_dir: str, filename: str) -> str:
    """Resolve relative mosaic filenames as the old script working directory did."""
    if os.path.isabs(filename):
        return filename
    return os.path.join(pipeline_working_dir, filename)


def run_mosaic_image_type(
    image_type: MosaicImageTypePayload,
    compress_images: bool,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run mosaicking for one image type and return a file output record."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    sector_images = image_type["sector_image_filenames"]
    sector_vertices = image_type["sector_vertices_filenames"]
    regridded_images = image_type["regridded_image_filenames"]
    template_path = image_type["template_image_path"]
    mosaic_filename = image_type["mosaic_filename"]
    mosaic_path = image_type["mosaic_path"]

    resolved_sector_images = [_work_path(pipeline_working_dir, image) for image in sector_images]
    resolved_sector_vertices = [
        _work_path(pipeline_working_dir, vertices) for vertices in sector_vertices
    ]
    resolved_regridded_images = [
        _work_path(pipeline_working_dir, image) for image in regridded_images
    ]

    make_mosaic_template(
        resolved_sector_images,
        resolved_sector_vertices,
        template_path,
    )
    require_file(template_path, "Mosaic output")
    for sector_image, vertices_file, regridded_image in zip(
        resolved_sector_images, resolved_sector_vertices, resolved_regridded_images
    ):
        regrid_image(
            sector_image,
            template_path,
            vertices_file,
            regridded_image,
        )
        require_file(regridded_image, "Mosaic output")
    make_mosaic(resolved_regridded_images, template_path, mosaic_path)
    output_record = require_file(mosaic_path, "Mosaic output")
    if compress_images:
        compressed_path = f"{mosaic_path}.fz"
        output_record = _run_command_and_require_file(
            build_compress_mosaic_command(mosaic_filename),
            compressed_path,
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )
        publish_fits_image_artifacts([output_record], pipeline_working_dir)
        return output_record
    publish_fits_image_artifacts([output_record], pipeline_working_dir)
    return output_record


@task(name="mosaic_image_type")
def mosaic_image_type_task(
    image_type: MosaicImageTypePayload,
    compress_images: bool,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    """Prefect task wrapper for one image type."""
    with publish_python_logs_to_prefect():
        return run_mosaic_image_type(
            image_type,
            compress_images,
            pipeline_working_dir,
            execution_config=execution_config,
        )


def _result_from_mosaic_records(outputs: list[dict]) -> dict:
    result = {"mosaic_image": outputs}
    validate_output_record(result["mosaic_image"])
    return result


def _run_mosaic_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    assert_serializable_payload(payload)
    payload = validate_mosaic_payload(payload)
    if payload["skip_processing"]:
        return {}

    config = execution_config or ExecutionConfig(task_runner="sync")
    operation_name = operation_run_name(payload, "mosaic")
    outputs = [
        mosaic_image_type_task.with_options(
            task_run_name=task_run_name(operation_name, "image_type", index + 1)
        ).submit(
            image_type,
            payload["compress_images"],
            payload["pipeline_working_dir"],
            execution_config=config,
        )
        for index, image_type in enumerate(payload["image_types"])
    ]
    outputs = [output.result() for output in outputs]
    return _result_from_mosaic_records(outputs)


@flow(name="mosaic")
def _mosaic_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect implementation for Mosaic."""
    with publish_python_logs_to_prefect():
        return _run_mosaic_prefect_tasks(payload, execution_config=execution_config)


def mosaic_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
):
    """Prefect entry point for Mosaic."""
    return run_flow_with_task_runner(
        _mosaic_flow,
        payload,
        flow_run_name=operation_run_name(payload, "mosaic"),
        execution_config=execution_config,
    )
