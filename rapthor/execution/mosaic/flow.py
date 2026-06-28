"""Prefect flow for the Mosaic operation."""

import os
from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.artifacts import publish_fits_image_artifacts
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.mosaic.commands import (
    build_compress_mosaic_command,
    build_make_mosaic_command,
    build_make_mosaic_template_command,
    build_regrid_image_command,
)
from rapthor.execution.mosaic.payloads import MosaicImageTypePayload, validate_mosaic_payload
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.shell import ShellCommand, run_shell_command
from rapthor.execution.task_runner import run_flow_with_task_runner
from rapthor.lib.records import file_record, validate_output_record


def _run_shell_and_validate(
    command: list[str],
    output_path: str,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> None:
    run_shell_command(
        ShellCommand(command=command, working_directory=pipeline_working_dir),
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    if not os.path.isfile(output_path):
        raise FileNotFoundError(f"Mosaic output was not created: {output_path}")


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
    template_filename = image_type["template_image_filename"]
    template_path = image_type["template_image_path"]
    mosaic_filename = image_type["mosaic_filename"]
    mosaic_path = image_type["mosaic_path"]

    _run_shell_and_validate(
        build_make_mosaic_template_command(sector_images, sector_vertices, template_filename),
        template_path,
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )
    for sector_image, vertices_file, regridded_image in zip(
        sector_images, sector_vertices, regridded_images
    ):
        _run_shell_and_validate(
            build_regrid_image_command(
                sector_image,
                template_filename,
                vertices_file,
                regridded_image,
            ),
            os.path.join(pipeline_working_dir, regridded_image),
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )
    _run_shell_and_validate(
        build_make_mosaic_command(regridded_images, template_filename, mosaic_filename),
        mosaic_path,
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )
    if compress_images:
        compressed_path = f"{mosaic_path}.fz"
        _run_shell_and_validate(
            build_compress_mosaic_command(mosaic_filename),
            compressed_path,
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )
        output_record = file_record(compressed_path)
        publish_fits_image_artifacts([output_record], pipeline_working_dir)
        return output_record
    output_record = file_record(mosaic_path)
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


def run_mosaic_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run mosaic commands and return finalizer-compatible outputs."""
    assert_serializable_payload(payload)
    payload = validate_mosaic_payload(payload)
    if payload["skip_processing"]:
        return {}

    config = execution_config or ExecutionConfig(task_runner="sync")
    outputs = [
        run_mosaic_image_type(
            image_type,
            payload["compress_images"],
            payload["pipeline_working_dir"],
            execution_config=config,
            shell_operation_cls=shell_operation_cls,
        )
        for image_type in payload["image_types"]
    ]
    return _result_from_mosaic_records(outputs)


def _run_mosaic_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    payload = validate_mosaic_payload(payload)
    if payload["skip_processing"]:
        return {}

    config = execution_config or ExecutionConfig(task_runner="sync")
    outputs = [
        mosaic_image_type_task.submit(
            image_type,
            payload["compress_images"],
            payload["pipeline_working_dir"],
            execution_config=config,
        )
        for image_type in payload["image_types"]
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
        execution_config=execution_config,
    )
