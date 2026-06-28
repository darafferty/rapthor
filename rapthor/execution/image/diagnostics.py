"""Image-sector diagnostics helpers."""

import os
from typing import Mapping, Optional

from rapthor.execution.artifacts import publish_plot_file_records
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.image.commands import build_calculate_image_diagnostics_command
from rapthor.execution.image.payloads import ImageSectorPayload
from rapthor.execution.outputs import file_records_for_patterns, require_file
from rapthor.execution.shell import run_external_command
from rapthor.lib.records import file_record


def run_image_diagnostics(
    sector: ImageSectorPayload,
    image_name: str,
    nonpb_image: Mapping[str, str],
    pb_image: Mapping[str, str],
    flat_noise_rms: Mapping[str, str],
    true_sky_rms: Mapping[str, str],
    source_catalog: Mapping[str, str],
    diagnostics: Mapping[str, str],
    region_record: Optional[Mapping[str, str]],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> tuple[dict, Optional[dict], list[dict]]:
    """Calculate and publish image diagnostics for a sector."""
    diagnostics_command = build_calculate_image_diagnostics_command(
        nonpb_image["path"],
        flat_noise_rms["path"],
        pb_image["path"],
        true_sky_rms["path"],
        source_catalog["path"],
        list(sector["obs_original_paths"]),
        list(sector["obs_starttime"]),
        list(sector["obs_ntimes"]),
        diagnostics["path"],
        image_name,
        bool(sector["allow_internet_access"]),
        facet_region_file=None if region_record is None else region_record["path"],
        photometry_skymodel=sector.get("photometry_skymodel"),
        astrometry_skymodel=sector.get("astrometry_skymodel"),
    )
    run_external_command(
        diagnostics_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    diagnostics = require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.image_diagnostics.json"),
        "Image diagnostics",
    )
    offsets_path = os.path.join(pipeline_working_dir, f"{image_name}.astrometry_offsets.json")
    offsets = file_record(offsets_path) if os.path.isfile(offsets_path) else None
    diagnostic_plots = file_records_for_patterns(
        [os.path.join(pipeline_working_dir, f"{image_name}*.pdf")]
    )
    publish_plot_file_records([diagnostics], pipeline_working_dir)
    publish_plot_file_records(diagnostic_plots, pipeline_working_dir)
    return diagnostics, offsets, diagnostic_plots
