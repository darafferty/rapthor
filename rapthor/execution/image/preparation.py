"""Prepare image-sector inputs before WSClean runs."""

import os
from typing import Optional

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.image.commands import (
    PrepareImagingDataOptions,
    build_concat_time_command,
    build_prepare_imaging_data_command,
)
from rapthor.execution.image.masking import blank_image
from rapthor.execution.image.payloads import ImageSectorPayload
from rapthor.execution.outputs import require_directory, require_file
from rapthor.execution.regions import make_ds9_region_from_skymodel
from rapthor.execution.shell import run_external_command


def prepare_and_concatenate_visibilities(
    sector: ImageSectorPayload,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> tuple[list[dict], dict]:
    """Prepare per-observation imaging MSs and concatenate them for imaging."""
    prepared_records = []
    for prepare_task in sector["prepare_tasks"]:
        if not os.path.isdir(str(prepare_task["msout_path"])):
            command = build_prepare_imaging_data_command(
                PrepareImagingDataOptions(
                    msin=str(prepare_task["msin"]),
                    data_colname=str(sector["data_colname"]),
                    msout=str(prepare_task["msout"]),
                    starttime=str(prepare_task["starttime"]),
                    ntimes=int(prepare_task["ntimes"]),
                    phasecenter=str(sector["phasecenter"]),
                    freqstep=int(prepare_task["freqstep"]),
                    timestep=int(prepare_task["timestep"]),
                    beamdir=str(sector["phasecenter"]),
                    num_threads=int(sector["max_threads"]),
                    steps=str(sector["prepare_data_steps"]),
                    maxinterval=prepare_task.get("maxinterval"),
                    timebase=sector.get("timebase"),
                    h5parm=sector.get("h5parm"),
                    fulljones_h5parm=sector.get("fulljones_h5parm"),
                    normalize_h5parm=sector.get("input_normalize_h5parm"),
                    central_patch_name=sector.get("central_patch_name"),
                    applycal_steps=sector.get("prepare_data_applycal_steps"),
                )
            )
            run_external_command(
                command,
                pipeline_working_dir,
                execution_config,
                shell_operation_cls=shell_operation_cls,
            )
        prepared_records.append(
            require_directory(str(prepare_task["msout_path"]), "Prepared imaging MS")
        )

    prepared_paths = [record["path"] for record in prepared_records]
    if not os.path.isdir(str(sector["concat_path"])):
        concat_command = build_concat_time_command(
            prepared_paths, str(sector["concat_filename"]), str(sector["data_colname"])
        )
        run_external_command(
            concat_command,
            pipeline_working_dir,
            execution_config,
            shell_operation_cls=shell_operation_cls,
        )
    concat_record = require_directory(str(sector["concat_path"]), "Concatenated imaging MS")
    return prepared_records, concat_record


def ensure_imaging_mask(
    sector: ImageSectorPayload,
) -> dict:
    """Create the clean mask if needed and return its output record."""
    if not os.path.isfile(str(sector["mask_path"])):
        blank_image(
            str(sector["mask_path"]),
            input_image=sector.get("previous_mask_filename"),
            vertices_file=str(sector["vertices_file"]),
            reference_ra_deg=float(sector["ra"]),
            reference_dec_deg=float(sector["dec"]),
            cellsize_deg=float(sector["cellsize_deg"]),
            imsize=list(sector["wsclean_imsize"]),
        )
    return require_file(str(sector["mask_path"]), "Imaging mask")


def ensure_facet_region(
    sector: ImageSectorPayload,
) -> Optional[dict]:
    """Create the facet region file when facet imaging is enabled."""
    if not sector["use_facets"]:
        return None
    if not os.path.isfile(str(sector["facet_region_path"])):
        make_ds9_region_from_skymodel(
            str(sector["facet_skymodel"]),
            float(sector["ra_mid"]),
            float(sector["dec_mid"]),
            float(sector["width_ra"]),
            float(sector["width_dec"]),
            str(sector["facet_region_path"]),
        )
    return require_file(str(sector["facet_region_path"]), "Facet region file")
