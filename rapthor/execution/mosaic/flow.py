"""Prefect flow for the Mosaic operation."""

import logging
import os
import re
from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.artifacts import publish_fits_image_artifacts
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.mosaic.commands import build_compress_mosaic_command
from rapthor.execution.mosaic.images import (
    is_sparse_model_product,
    make_mosaic,
    make_mosaic_template,
    regrid_image,
    regrid_sparse_model_image,
)
from rapthor.execution.mosaic.model_rendering import render_model_mosaic_with_wsclean
from rapthor.execution.mosaic.payloads import MosaicProductPayload, validate_mosaic_payload
from rapthor.execution.outputs import require_file
from rapthor.execution.payloads import assert_serializable_payload
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.run_names import operation_run_name, task_run_name
from rapthor.execution.shell import run_external_command
from rapthor.execution.task_runner import run_flow_with_task_runner
from rapthor.lib.records import file_record_path, validate_output_record

log = logging.getLogger("rapthor:mosaic")

_FITS_SUFFIX = re.compile(r"\.fits(?:\.fz)?$")
_MOSAIC_OUTPUT_PREFIX = re.compile(r"^mosaic_\d+-")
_MOSAIC_PRODUCT_LABEL_SEPARATOR = re.compile(r"[^A-Za-z0-9]+")


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


def _finalize_mosaic_output(
    output_record: dict,
    mosaic_filename: str,
    mosaic_path: str,
    compress_images: bool,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    """Compress and/or publish one mosaic output record."""
    if compress_images:
        compressed_path = f"{mosaic_path}.fz"
        output_record = _run_command_and_require_file(
            build_compress_mosaic_command(mosaic_filename),
            compressed_path,
            pipeline_working_dir,
            execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    if execution_config.publish_fits_previews:
        publish_fits_image_artifacts(
            [output_record],
            pipeline_working_dir,
            clip_percentile=execution_config.fits_preview_clip_percentile,
        )
    return output_record


def _work_path(pipeline_working_dir: str, filename: str) -> str:
    """Resolve relative mosaic filenames as the old script working directory did."""
    if os.path.isabs(filename):
        return filename
    return os.path.join(pipeline_working_dir, filename)


def run_mosaic_product(
    mosaic_product: MosaicProductPayload,
    compress_images: bool,
    pipeline_working_dir: str,
    template_record: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run mosaicking for one output product and return a file output record."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    sector_images = mosaic_product["sector_image_filenames"]
    sector_vertices = mosaic_product["sector_vertices_filenames"]
    sector_model_skymodels = mosaic_product["sector_model_skymodel_filenames"]
    regridded_images = mosaic_product["regridded_image_filenames"]
    template_path = file_record_path(template_record)
    mosaic_filename = mosaic_product["mosaic_filename"]
    mosaic_path = mosaic_product["mosaic_path"]

    resolved_sector_images = [_work_path(pipeline_working_dir, image) for image in sector_images]
    resolved_sector_vertices = [
        _work_path(pipeline_working_dir, vertices) for vertices in sector_vertices
    ]
    resolved_regridded_images = [
        _work_path(pipeline_working_dir, image) for image in regridded_images
    ]

    log.info(
        "Mosaicking %s from %d sector image(s)",
        mosaic_filename,
        len(resolved_sector_images),
    )
    log.info(
        "Mosaic sector inputs for %s: %s",
        mosaic_filename,
        ", ".join(os.path.basename(image) for image in resolved_sector_images),
    )
    require_file(template_path, "Mosaic output")
    if sector_model_skymodels and is_sparse_model_product(mosaic_filename):
        resolved_sector_skymodels = [
            _work_path(pipeline_working_dir, skymodel) for skymodel in sector_model_skymodels
        ]
        log.info(
            "Rendering model mosaic %s with WSClean from %d sector sky model(s)",
            mosaic_filename,
            len(resolved_sector_skymodels),
        )
        output_record = render_model_mosaic_with_wsclean(
            resolved_sector_skymodels,
            template_path,
            mosaic_path,
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )
        return _finalize_mosaic_output(
            output_record,
            mosaic_filename,
            mosaic_path,
            compress_images,
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )

    regrid = regrid_sparse_model_image if is_sparse_model_product(mosaic_filename) else regrid_image
    for sector_image, vertices_file, regridded_image in zip(
        resolved_sector_images, resolved_sector_vertices, resolved_regridded_images
    ):
        regrid(
            sector_image,
            template_path,
            vertices_file,
            regridded_image,
        )
        require_file(regridded_image, "Mosaic output")
    make_mosaic(resolved_regridded_images, template_path, mosaic_path)
    output_record = require_file(mosaic_path, "Mosaic output")
    return _finalize_mosaic_output(
        output_record,
        mosaic_filename,
        mosaic_path,
        compress_images,
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )


def run_mosaic_template(
    mosaic_product: MosaicProductPayload,
    pipeline_working_dir: str,
) -> dict:
    """Build the shared mosaic template once for all mosaic products."""
    sector_images = mosaic_product["sector_image_filenames"]
    sector_vertices = mosaic_product["sector_vertices_filenames"]
    template_path = mosaic_product["template_image_path"]
    resolved_sector_images = [_work_path(pipeline_working_dir, image) for image in sector_images]
    resolved_sector_vertices = [
        _work_path(pipeline_working_dir, vertices) for vertices in sector_vertices
    ]

    log.info(
        "Building mosaic template %s from %d sector image(s)",
        os.path.basename(template_path),
        len(resolved_sector_images),
    )
    make_mosaic_template(
        resolved_sector_images,
        resolved_sector_vertices,
        template_path,
    )
    return require_file(template_path, "Mosaic output")


@task(name="make_mosaic_template")
def mosaic_template_task(
    mosaic_product: MosaicProductPayload,
    pipeline_working_dir: str,
) -> dict:
    """Prefect task wrapper for the shared mosaic template."""
    with publish_python_logs_to_prefect():
        return run_mosaic_template(mosaic_product, pipeline_working_dir)


@task(name="mosaic")
def mosaic_task(
    mosaic_product: MosaicProductPayload,
    compress_images: bool,
    pipeline_working_dir: str,
    template_record: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    """Prefect task wrapper for one mosaic product."""
    with publish_python_logs_to_prefect():
        return run_mosaic_product(
            mosaic_product,
            compress_images,
            pipeline_working_dir,
            template_record,
            execution_config=execution_config,
        )


def _result_from_mosaic_records(outputs: list[dict]) -> dict:
    result = {"mosaic_image": outputs}
    validate_output_record(result["mosaic_image"])
    return result


def _mosaic_product_label(mosaic_filename: str) -> str:
    """Return a stable task-label from a mosaic output filename."""
    label = os.path.basename(mosaic_filename)
    label = _FITS_SUFFIX.sub("", label)
    label = _MOSAIC_OUTPUT_PREFIX.sub("", label)
    return _MOSAIC_PRODUCT_LABEL_SEPARATOR.sub("_", label).strip("_")


def _mosaic_task_run_name(mosaic_product: MosaicProductPayload, index: int) -> str:
    """Return a compact task-run label for one mosaic product."""
    label = _mosaic_product_label(mosaic_product["mosaic_filename"])
    return task_run_name("mosaic", label or index + 1)


def _run_mosaic_prefect_tasks(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
) -> dict:
    assert_serializable_payload(payload)
    payload = validate_mosaic_payload(payload)
    if payload["skip_processing"]:
        return {}

    config = execution_config or ExecutionConfig(task_runner="sync")
    template_record = mosaic_template_task.with_options(
        task_run_name="make_mosaic_template"
    ).submit(
        payload["mosaic_products"][0],
        payload["pipeline_working_dir"],
    )
    outputs = [
        mosaic_task.with_options(task_run_name=_mosaic_task_run_name(mosaic_product, index)).submit(
            mosaic_product,
            payload["compress_images"],
            payload["pipeline_working_dir"],
            template_record,
            execution_config=config,
        )
        for index, mosaic_product in enumerate(payload["mosaic_products"])
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
