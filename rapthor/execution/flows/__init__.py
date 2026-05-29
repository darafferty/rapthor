"""Operation-level Prefect flows."""

from rapthor.execution.flows.concatenate import (
    build_concatenate_command,
    concatenate_epoch_task,
    concatenate_flow,
    concatenate_payload_from_inputs,
    normalized_concatenate_command,
    run_concatenate_epoch,
    run_concatenate_flow,
)
from rapthor.execution.flows.mosaic import (
    build_compress_mosaic_command,
    build_make_mosaic_command,
    build_make_mosaic_template_command,
    build_regrid_image_command,
    mosaic_flow,
    mosaic_image_type_task,
    mosaic_payload_from_inputs,
    normalized_make_mosaic_command,
    normalized_make_mosaic_template_command,
    normalized_regrid_image_command,
    run_mosaic_flow,
    run_mosaic_image_type,
)

__all__ = [
    "build_concatenate_command",
    "build_compress_mosaic_command",
    "build_make_mosaic_command",
    "build_make_mosaic_template_command",
    "build_regrid_image_command",
    "concatenate_epoch_task",
    "concatenate_flow",
    "concatenate_payload_from_inputs",
    "mosaic_flow",
    "mosaic_image_type_task",
    "mosaic_payload_from_inputs",
    "normalized_concatenate_command",
    "normalized_make_mosaic_command",
    "normalized_make_mosaic_template_command",
    "normalized_regrid_image_command",
    "run_concatenate_epoch",
    "run_concatenate_flow",
    "run_mosaic_flow",
    "run_mosaic_image_type",
]
