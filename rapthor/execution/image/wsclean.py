"""WSClean execution helpers for one image sector."""

import os
from typing import Mapping, Optional

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.image.beam import ensure_image_beam
from rapthor.execution.image.commands import (
    ATERM_CONFIG_FILENAME,
    WscleanFacetOptions,
    WscleanOptions,
    WscleanScreenOptions,
    build_aterm_config_content,
    build_wsclean_facets_command,
    build_wsclean_mpi_facets_command,
    build_wsclean_mpi_no_dde_command,
    build_wsclean_mpi_screens_command,
    build_wsclean_no_dde_command,
    build_wsclean_restore_command,
    build_wsclean_screens_command,
)
from rapthor.execution.image.outputs import mfs_non_pb_image_patterns, mfs_pb_image_patterns
from rapthor.execution.image.payloads import ImageSectorPayload
from rapthor.execution.outputs import (
    cleanup_directory,
    first_existing_file,
    optional_first_existing_file,
    require_file,
)
from rapthor.execution.resources import (
    ResourceRequest,
    thread_environment,
    validate_resource_request,
)
from rapthor.execution.shell import run_external_command


def _restore_bright_source_image(
    image_record: Mapping[str, str],
    bright_skymodel_pb: str,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    numthreads: int,
    description: str,
    shell_operation_cls=None,
) -> dict:
    output_image = os.path.basename(image_record["path"])
    command = build_wsclean_restore_command(
        image_record["path"], bright_skymodel_pb, output_image, numthreads
    )
    run_external_command(
        command, pipeline_working_dir, execution_config, shell_operation_cls=shell_operation_cls
    )
    return require_file(os.path.join(pipeline_working_dir, output_image), description)


def _write_aterm_config(pipeline_working_dir: str, h5parm: str) -> str:
    config_path = os.path.join(pipeline_working_dir, ATERM_CONFIG_FILENAME)
    os.makedirs(pipeline_working_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(build_aterm_config_content(h5parm))
    return config_path


def _mpi_environment(
    threads: int,
    processes: int,
    execution_config: ExecutionConfig,
) -> Mapping[str, str]:
    resource_request = validate_resource_request(
        ResourceRequest(
            name="wsclean-mpi",
            threads=threads,
            processes=processes,
            use_mpi=True,
            exclusive=True,
        ),
        execution_config,
    )
    return thread_environment(resource_request)


def _wsclean_threads_for_sector(sector: ImageSectorPayload) -> int:
    if sector["use_mpi"]:
        return int(sector["mpi_cpus_per_task"])
    return int(sector["max_threads"])


def _wsclean_environment_for_sector(
    sector: ImageSectorPayload,
    execution_config: ExecutionConfig,
) -> Optional[Mapping[str, str]]:
    if not sector["use_mpi"]:
        return None
    return _mpi_environment(
        _wsclean_threads_for_sector(sector),
        int(sector["mpi_nnodes"]),
        execution_config,
    )


def _select_wsclean_command_for_sector(
    sector: ImageSectorPayload,
    concat_record: Mapping[str, str],
    mask_record: Mapping[str, str],
    region_record: Optional[Mapping[str, str]],
    temp_dir: str,
) -> list[str]:
    common_options = WscleanOptions(
        msin=concat_record["path"],
        name=str(sector["image_name"]),
        mask=mask_record["path"],
        imsize=list(sector["wsclean_imsize"]),
        niter=int(sector["wsclean_niter"]),
        nmiter=int(sector["wsclean_nmiter"]),
        robust=float(sector["robust"]),
        min_uv_lambda=float(sector["min_uv_lambda"]),
        max_uv_lambda=float(sector["max_uv_lambda"]),
        mgain=float(sector["mgain"]),
        multiscale=bool(sector["do_multiscale"]),
        save_source_list=bool(sector["save_source_list"]),
        pol=str(sector["pol"]),
        link_polarizations=(
            str(sector["link_polarizations"]) if sector["link_polarizations"] else None
        ),
        join_polarizations=bool(sector["join_polarizations"]),
        skip_final_iteration=bool(sector["skip_final_iteration"]),
        cellsize_deg=float(sector["cellsize_deg"]),
        channels_out=int(sector["channels_out"]),
        deconvolution_channels=int(sector["deconvolution_channels"]),
        fit_spectral_pol=int(sector["fit_spectral_pol"]),
        taper_arcsec=float(sector["taper_arcsec"]),
        local_rms_strength=float(sector["local_rms_strength"]),
        local_rms_window=float(sector["local_rms_window"]),
        local_rms_method=str(sector["local_rms_method"]),
        memory_gb=float(sector["wsclean_mem"]),
        auto_mask=float(sector["auto_mask"]),
        auto_mask_nmiter=int(sector["auto_mask_nmiter"]),
        idg_mode=str(sector["idg_mode"]),
        num_threads=_wsclean_threads_for_sector(sector),
        num_deconvolution_threads=int(sector["deconvolution_threads"]),
        num_gridding_tasks=int(sector["parallel_gridding_tasks"]),
        dd_psf_grid=list(sector["dd_psf_grid"]),
        apply_time_frequency_smearing=bool(sector["apply_time_frequency_smearing"]),
        temp_dir=temp_dir,
        update_model_required=bool(sector["make_residual_visibilities"]),
    )
    if sector["use_facets"]:
        if region_record is None:
            raise ValueError("Facet imaging requires a facet region record")
        facet_options = WscleanFacetOptions(
            common=common_options,
            scalar_visibilities=bool(sector["scalar_visibilities"]),
            diagonal_visibilities=bool(sector["diagonal_visibilities"]),
            h5parm=str(sector["h5parm"]),
            soltabs=str(sector["soltabs"]),
            region_file=region_record["path"],
            num_gridding_tasks=int(sector["parallel_gridding_tasks"]),
            shared_facet_reads=bool(sector["shared_facet_reads"]),
            shared_facet_writes=bool(sector["shared_facet_writes"]),
        )
        if sector["use_mpi"]:
            return build_wsclean_mpi_facets_command(
                mpi_nnodes=int(sector["mpi_nnodes"]), options=facet_options
            )
        return build_wsclean_facets_command(facet_options)
    if sector["apply_screens"]:
        screen_options = WscleanScreenOptions(
            common=common_options,
            interval=list(sector["interval"]),
        )
        if sector["use_mpi"]:
            return build_wsclean_mpi_screens_command(
                mpi_nnodes=int(sector["mpi_nnodes"]), options=screen_options
            )
        return build_wsclean_screens_command(screen_options)
    if sector["use_mpi"]:
        return build_wsclean_mpi_no_dde_command(
            mpi_nnodes=int(sector["mpi_nnodes"]), options=common_options
        )
    return build_wsclean_no_dde_command(common_options)


def run_or_reuse_wsclean_images(
    sector: ImageSectorPayload,
    concat_record: Mapping[str, str],
    mask_record: Mapping[str, str],
    region_record: Optional[Mapping[str, str]],
    image_name: str,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> tuple[dict, dict, bool]:
    """Run WSClean when required image products are missing."""
    nonpb_image_patterns = mfs_non_pb_image_patterns(image_name, pipeline_working_dir)
    pb_image_patterns = mfs_pb_image_patterns(image_name, pipeline_working_dir)
    nonpb_image = optional_first_existing_file(nonpb_image_patterns)
    pb_image = optional_first_existing_file(pb_image_patterns)
    residual_missing = False
    if sector["make_residual_visibilities"]:
        if sector["residual_path"] is None:
            raise ValueError("Residual visibility path is required when residuals are requested")
        residual_missing = not os.path.isdir(str(sector["residual_path"]))
    if nonpb_image is not None and pb_image is not None and not residual_missing:
        return nonpb_image, pb_image, False

    if sector["apply_screens"]:
        _write_aterm_config(pipeline_working_dir, str(sector["h5parm"]))

    temp_dir = os.path.join(pipeline_working_dir, f"{image_name}_wsclean_tmp")
    wsclean_command = _select_wsclean_command_for_sector(
        sector, concat_record, mask_record, region_record, temp_dir
    )
    wsclean_environment = _wsclean_environment_for_sector(sector, execution_config)
    try:
        os.makedirs(temp_dir, exist_ok=True)
        run_external_command(
            wsclean_command,
            pipeline_working_dir,
            execution_config,
            shell_operation_cls=shell_operation_cls,
            environment=wsclean_environment,
        )
    finally:
        cleanup_directory(temp_dir)

    return (
        first_existing_file(nonpb_image_patterns, "WSClean non-PB image"),
        first_existing_file(pb_image_patterns, "WSClean PB image"),
        True,
    )


def restore_bright_source_images(
    sector: ImageSectorPayload,
    nonpb_image: dict,
    pb_image: dict,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> tuple[dict, dict]:
    """Restore bright-source models into PB and non-PB images when requested."""
    if not sector["peel_bright_sources"]:
        return nonpb_image, pb_image
    pb_image = _restore_bright_source_image(
        pb_image,
        str(sector["bright_skymodel_pb"]),
        pipeline_working_dir,
        execution_config,
        int(sector["max_threads"]),
        "Bright-source restored PB image",
        shell_operation_cls=shell_operation_cls,
    )
    nonpb_image = _restore_bright_source_image(
        nonpb_image,
        str(sector["bright_skymodel_pb"]),
        pipeline_working_dir,
        execution_config,
        int(sector["max_threads"]),
        "Bright-source restored non-PB image",
        shell_operation_cls=shell_operation_cls,
    )
    return nonpb_image, pb_image


def check_wsclean_beams(
    image_records: tuple[dict, dict],
    sector: ImageSectorPayload,
) -> None:
    """Run beam checks for newly created WSClean images."""
    for image_record in image_records:
        ensure_image_beam(image_record["path"], float(sector["taper_arcsec"]))
        require_file(image_record["path"], "Beam-checked image")
