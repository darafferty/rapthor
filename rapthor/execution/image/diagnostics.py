"""Image-sector diagnostics helpers."""

import os
from typing import Mapping, Optional

from rapthor.execution.artifacts import publish_plot_file_records
from rapthor.execution.image.diagnostic_calculation import calculate_image_diagnostics
from rapthor.execution.image.payloads import ImageSectorPayload
from rapthor.execution.outputs import file_records_for_patterns, require_file
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
) -> tuple[dict, Optional[dict], list[dict]]:
    """Calculate and publish image diagnostics for a sector."""
    calculate_image_diagnostics(
        nonpb_image["path"],
        flat_noise_rms["path"],
        pb_image["path"],
        true_sky_rms["path"],
        source_catalog["path"],
        list(sector["obs_original_paths"]),
        list(sector["obs_starttime"]),
        list(sector["obs_ntimes"]),
        diagnostics["path"],
        os.path.join(pipeline_working_dir, image_name),
        allow_internet_access=bool(sector["allow_internet_access"]),
        facet_region_file=None if region_record is None else region_record["path"],
        photometry_comparison_skymodel=sector.get("photometry_skymodel"),
        astrometry_comparison_skymodel=sector.get("astrometry_skymodel"),
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
