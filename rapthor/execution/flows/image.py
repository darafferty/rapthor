"""Prefect flows for imaging."""

import glob
import os
import shutil
from typing import Mapping, Optional

from prefect import flow, task

from rapthor.execution.artifacts import publish_fits_image_artifacts, publish_plot_file_records
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.flows.runtime import run_flow_with_task_runner
from rapthor.execution.image_commands import (
    ATERM_CONFIG_FILENAME,
    build_aterm_config_content,
    build_blank_image_command,
    build_calculate_image_diagnostics_command,
    build_check_image_beam_command,
    build_compress_sector_images_command,
    build_concat_time_command,
    build_filter_skymodel_command,
    build_make_catalog_from_image_cube_command,
    build_make_image_cube_command,
    build_make_region_file_command,
    build_make_skymodel_image_command,
    build_normalize_flux_scale_command,
    build_prepare_imaging_data_command,
    build_wsclean_facets_command,
    build_wsclean_mpi_facets_command,
    build_wsclean_mpi_no_dde_command,
    build_wsclean_mpi_screens_command,
    build_wsclean_no_dde_command,
    build_wsclean_restore_command,
    build_wsclean_screens_command,
)
from rapthor.execution.image_payloads import validate_image_payload
from rapthor.execution.outputs import (
    directory_record,
    file_record,
    validate_output_record,
)
from rapthor.execution.payloads import (
    ImageCubeSpecPayload,
    ImageSectorPayload,
    assert_serializable_payload,
)
from rapthor.execution.prefect_logging import publish_python_logs_to_prefect
from rapthor.execution.resources import (
    ResourceRequest,
    thread_environment,
    validate_resource_request,
)
from rapthor.execution.shell import ShellCommand, run_shell_command


def _require_file(path: str, description: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} was not created: {path}")
    return file_record(path)


def _require_directory(path: str, description: str) -> dict:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{description} was not created: {path}")
    return directory_record(path)


def _first_existing_file(patterns: list[str], description: str) -> dict:
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path):
                return file_record(path)
    raise FileNotFoundError(f"{description} was not created: {', '.join(patterns)}")


def _optional_first_existing_file(patterns: list[str]) -> Optional[dict]:
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path):
                return file_record(path)
    return None


def _file_records_for_required_patterns(patterns: list[str], description: str) -> list[dict]:
    records = _file_records_for_patterns(patterns)
    if not records:
        raise FileNotFoundError(f"{description} was not created: {', '.join(patterns)}")
    return records


def _file_records_for_patterns(patterns: list[str]) -> list[dict]:
    records = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path):
                records.append(file_record(path))
    return records


def _compressed_file_record(record: dict, description: str) -> dict:
    return _require_file(f"{record['path']}.fz", description)


def _cleanup_directory(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)


def _compress_image_records(
    image_name: str,
    sector_images: list[dict],
    extra_images: list[dict],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> tuple[list[dict], list[dict]]:
    command = build_compress_sector_images_command(
        [record["path"] for record in sector_images + extra_images]
    )
    _run_shell(
        command, pipeline_working_dir, execution_config, shell_operation_cls=shell_operation_cls
    )
    compressed_sector_images = [
        _compressed_file_record(sector_images[0], "Compressed WSClean non-PB image"),
        _compressed_file_record(sector_images[1], "Compressed WSClean PB image"),
    ]
    compressed_extra_images = _file_records_for_patterns(
        [
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-[QUV]-image.fits.fz"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-[QUV]-image-pb.fits.fz"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-*residual.fits.fz"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-*model-pb.fits.fz"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-*dirty.fits.fz"),
        ]
    )
    return compressed_sector_images, compressed_extra_images


def _channel_image_patterns(image_name: str, stokes: str, pipeline_working_dir: str) -> list[str]:
    if stokes == "I":
        return [
            os.path.join(pipeline_working_dir, f"{image_name}-0???-image-pb.fits"),
            os.path.join(pipeline_working_dir, f"{image_name}-0???-I-image-pb.fits"),
        ]
    return [os.path.join(pipeline_working_dir, f"{image_name}-0???-{stokes}-image-pb.fits")]


def _make_image_cube_records(
    image_name: str,
    image_cube_specs: list[ImageCubeSpecPayload],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> tuple[list[dict], list[dict], list[dict]]:
    image_cubes = []
    image_cube_beams = []
    image_cube_frequencies = []
    for spec in image_cube_specs:
        stokes = str(spec["pol"])
        image_cube_filename = str(spec["filename"])
        channel_images = _file_records_for_required_patterns(
            _channel_image_patterns(image_name, stokes, pipeline_working_dir),
            f"WSClean Stokes-{stokes} channel images",
        )
        command = build_make_image_cube_command(
            [record["path"] for record in channel_images], image_cube_filename
        )
        _run_shell(
            command, pipeline_working_dir, execution_config, shell_operation_cls=shell_operation_cls
        )
        image_cube_path = os.path.join(pipeline_working_dir, image_cube_filename)
        image_cubes.append(_require_file(image_cube_path, f"Stokes-{stokes} image cube"))
        image_cube_beams.append(
            _require_file(f"{image_cube_path}_beams.txt", f"Stokes-{stokes} image cube beams")
        )
        image_cube_frequencies.append(
            _require_file(
                f"{image_cube_path}_frequencies.txt",
                f"Stokes-{stokes} image cube frequencies",
            )
        )
    return image_cubes, image_cube_beams, image_cube_frequencies


def _make_normalization_records(
    image_cube: dict,
    image_cube_beams: dict,
    image_cube_frequencies: dict,
    concat_record: dict,
    sector: ImageSectorPayload,
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> tuple[dict, dict]:
    catalog_command = build_make_catalog_from_image_cube_command(
        image_cube["path"],
        image_cube_beams["path"],
        image_cube_frequencies["path"],
        str(sector["output_source_catalog_filename"]),
        float(sector["threshisl"]),
        float(sector["threshpix"]),
        int(sector["max_threads"]),
    )
    _run_shell(
        catalog_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    source_catalog = _require_file(
        str(sector["output_source_catalog_path"]), "Normalization source catalog"
    )

    normalize_command = build_normalize_flux_scale_command(
        source_catalog["path"],
        concat_record["path"],
        str(sector["output_normalize_h5parm_filename"]),
    )
    _run_shell(
        normalize_command,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    normalize_h5parm = _require_file(
        str(sector["output_normalize_h5parm_path"]), "Flux-scale normalization h5parm"
    )
    return source_catalog, normalize_h5parm


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
    _run_shell(
        command, pipeline_working_dir, execution_config, shell_operation_cls=shell_operation_cls
    )
    return _require_file(os.path.join(pipeline_working_dir, output_image), description)


def _write_aterm_config(pipeline_working_dir: str, h5parm: str) -> str:
    config_path = os.path.join(pipeline_working_dir, ATERM_CONFIG_FILENAME)
    os.makedirs(pipeline_working_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(build_aterm_config_content(h5parm))
    return config_path


def _run_shell(
    command: list[str],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
    environment: Optional[Mapping[str, str]] = None,
) -> None:
    run_shell_command(
        ShellCommand(
            command=command,
            environment={} if environment is None else dict(environment),
            working_directory=pipeline_working_dir,
        ),
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )


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
    common_kwargs = {
        "msin": concat_record["path"],
        "name": str(sector["image_name"]),
        "mask": mask_record["path"],
        "wsclean_imsize": list(sector["wsclean_imsize"]),
        "wsclean_niter": int(sector["wsclean_niter"]),
        "wsclean_nmiter": int(sector["wsclean_nmiter"]),
        "robust": float(sector["robust"]),
        "min_uv_lambda": float(sector["min_uv_lambda"]),
        "max_uv_lambda": float(sector["max_uv_lambda"]),
        "mgain": float(sector["mgain"]),
        "multiscale": bool(sector["do_multiscale"]),
        "save_source_list": bool(sector["save_source_list"]),
        "pol": str(sector["pol"]),
        "link_polarizations": sector["link_polarizations"],
        "join_polarizations": bool(sector["join_polarizations"]),
        "skip_final_iteration": bool(sector["skip_final_iteration"]),
        "cellsize_deg": float(sector["cellsize_deg"]),
        "channels_out": int(sector["channels_out"]),
        "deconvolution_channels": int(sector["deconvolution_channels"]),
        "fit_spectral_pol": int(sector["fit_spectral_pol"]),
        "taper_arcsec": float(sector["taper_arcsec"]),
        "local_rms_strength": float(sector["local_rms_strength"]),
        "local_rms_window": float(sector["local_rms_window"]),
        "local_rms_method": str(sector["local_rms_method"]),
        "wsclean_mem": float(sector["wsclean_mem"]),
        "auto_mask": float(sector["auto_mask"]),
        "auto_mask_nmiter": int(sector["auto_mask_nmiter"]),
        "idg_mode": str(sector["idg_mode"]),
        "num_threads": _wsclean_threads_for_sector(sector),
        "num_deconvolution_threads": int(sector["deconvolution_threads"]),
        "dd_psf_grid": list(sector["dd_psf_grid"]),
        "apply_time_frequency_smearing": bool(sector["apply_time_frequency_smearing"]),
        "temp_dir": temp_dir,
    }
    if sector["use_facets"]:
        facet_kwargs = {
            **common_kwargs,
            "scalar_visibilities": bool(sector["scalar_visibilities"]),
            "diagonal_visibilities": bool(sector["diagonal_visibilities"]),
            "h5parm": str(sector["h5parm"]),
            "soltabs": str(sector["soltabs"]),
            "region_file": region_record["path"],
            "num_gridding_threads": int(sector["parallel_gridding_threads"]),
            "shared_facet_reads": bool(sector["shared_facet_reads"]),
            "shared_facet_writes": bool(sector["shared_facet_writes"]),
        }
        if sector["use_mpi"]:
            return build_wsclean_mpi_facets_command(
                mpi_nnodes=int(sector["mpi_nnodes"]), **facet_kwargs
            )
        return build_wsclean_facets_command(**facet_kwargs)
    if sector["apply_screens"]:
        screen_kwargs = {
            **common_kwargs,
            "interval": list(sector["interval"]),
        }
        if sector["use_mpi"]:
            return build_wsclean_mpi_screens_command(
                mpi_nnodes=int(sector["mpi_nnodes"]), **screen_kwargs
            )
        return build_wsclean_screens_command(**screen_kwargs)
    if sector["use_mpi"]:
        return build_wsclean_mpi_no_dde_command(
            mpi_nnodes=int(sector["mpi_nnodes"]), **common_kwargs
        )
    return build_wsclean_no_dde_command(**common_kwargs)


def run_image_sector(
    sector: ImageSectorPayload,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run one imaging sector."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    prepared_records = []
    for prepare_task in sector["prepare_tasks"]:
        if not os.path.isdir(str(prepare_task["msout_path"])):
            command = build_prepare_imaging_data_command(
                str(prepare_task["msin"]),
                str(sector["data_colname"]),
                str(prepare_task["msout"]),
                str(prepare_task["starttime"]),
                int(prepare_task["ntimes"]),
                str(sector["phasecenter"]),
                int(prepare_task["freqstep"]),
                int(prepare_task["timestep"]),
                str(sector["phasecenter"]),
                int(sector["max_threads"]),
                str(sector["prepare_data_steps"]),
                maxinterval=prepare_task.get("maxinterval"),
                timebase=sector.get("timebase"),
                h5parm=sector.get("h5parm"),
                fulljones_h5parm=sector.get("fulljones_h5parm"),
                normalize_h5parm=sector.get("input_normalize_h5parm"),
                central_patch_name=sector.get("central_patch_name"),
                applycal_steps=sector.get("prepare_data_applycal_steps"),
            )
            _run_shell(
                command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls
            )
        prepared_records.append(
            _require_directory(str(prepare_task["msout_path"]), "Prepared imaging MS")
        )

    prepared_paths = [record["path"] for record in prepared_records]
    if not os.path.isdir(str(sector["concat_path"])):
        concat_command = build_concat_time_command(
            prepared_paths, str(sector["concat_filename"]), str(sector["data_colname"])
        )
        _run_shell(
            concat_command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls
        )
    concat_record = _require_directory(str(sector["concat_path"]), "Concatenated imaging MS")

    if not os.path.isfile(str(sector["mask_path"])):
        mask_command = build_blank_image_command(
            str(sector["mask_filename"]),
            list(sector["wsclean_imsize"]),
            str(sector["vertices_file"]),
            float(sector["ra"]),
            float(sector["dec"]),
            float(sector["cellsize_deg"]),
            image_filename=sector.get("previous_mask_filename"),
            region_file=sector.get("region_file"),
        )
        _run_shell(
            mask_command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls
        )
    mask_record = _require_file(str(sector["mask_path"]), "Imaging mask")

    region_record = None
    if sector["use_facets"]:
        if not os.path.isfile(str(sector["facet_region_path"])):
            region_command = build_make_region_file_command(
                str(sector["facet_skymodel"]),
                float(sector["ra_mid"]),
                float(sector["dec_mid"]),
                float(sector["width_ra"]),
                float(sector["width_dec"]),
                str(sector["facet_region_filename"]),
            )
            _run_shell(
                region_command,
                pipeline_working_dir,
                config,
                shell_operation_cls=shell_operation_cls,
            )
        region_record = _require_file(str(sector["facet_region_path"]), "Facet region file")

    temp_dir = os.path.join(pipeline_working_dir, f"{sector['image_name']}_wsclean_tmp")
    image_name = str(sector["image_name"])
    nonpb_image_patterns = [
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-image.fits"),
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-I-image.fits"),
    ]
    pb_image_patterns = [
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-image-pb.fits"),
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-I-image-pb.fits"),
    ]
    nonpb_image = _optional_first_existing_file(nonpb_image_patterns)
    pb_image = _optional_first_existing_file(pb_image_patterns)
    wsclean_ran = False
    if nonpb_image is None or pb_image is None:
        if sector["apply_screens"]:
            _write_aterm_config(pipeline_working_dir, str(sector["h5parm"]))
        wsclean_command = _select_wsclean_command_for_sector(
            sector, concat_record, mask_record, region_record, temp_dir
        )
        wsclean_environment = _wsclean_environment_for_sector(sector, config)
        try:
            os.makedirs(temp_dir, exist_ok=True)
            _run_shell(
                wsclean_command,
                pipeline_working_dir,
                config,
                shell_operation_cls=shell_operation_cls,
                environment=wsclean_environment,
            )
            wsclean_ran = True
        finally:
            _cleanup_directory(temp_dir)
        nonpb_image = _first_existing_file(nonpb_image_patterns, "WSClean non-PB image")
        pb_image = _first_existing_file(pb_image_patterns, "WSClean PB image")
    extra_images = _file_records_for_patterns(
        [
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-[QUV]-image.fits"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-[QUV]-image-pb.fits"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-*residual.fits"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-*model-pb.fits"),
            os.path.join(pipeline_working_dir, f"{image_name}-MFS-*dirty.fits"),
        ]
    )
    sector_images = [nonpb_image, pb_image]
    if sector["peel_bright_sources"]:
        pb_image = _restore_bright_source_image(
            pb_image,
            str(sector["bright_skymodel_pb"]),
            pipeline_working_dir,
            config,
            int(sector["max_threads"]),
            "Bright-source restored PB image",
            shell_operation_cls=shell_operation_cls,
        )
        nonpb_image = _restore_bright_source_image(
            nonpb_image,
            str(sector["bright_skymodel_pb"]),
            pipeline_working_dir,
            config,
            int(sector["max_threads"]),
            "Bright-source restored non-PB image",
            shell_operation_cls=shell_operation_cls,
        )
        sector_images = [nonpb_image, pb_image]
    image_cubes = []
    image_cube_beams = []
    image_cube_frequencies = []
    if sector["make_image_cube"]:
        image_cubes, image_cube_beams, image_cube_frequencies = _make_image_cube_records(
            image_name,
            list(sector["image_cube_specs"]),
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )

    skymodel_nonpb = None
    skymodel_pb = None
    if sector["save_source_list"]:
        skymodel_nonpb = _require_file(
            os.path.join(pipeline_working_dir, f"{image_name}-sources.txt"),
            "WSClean apparent-sky source list",
        )
        skymodel_pb = _require_file(
            os.path.join(pipeline_working_dir, f"{image_name}-sources-pb.txt"),
            "WSClean true-sky source list",
        )

    if wsclean_ran:
        for image_record in (pb_image, nonpb_image):
            command = build_check_image_beam_command(
                image_record["path"], float(sector["taper_arcsec"])
            )
            _run_shell(
                command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls
            )
            _require_file(image_record["path"], "Beam-checked image")

    filter_command = build_filter_skymodel_command(
        nonpb_image["path"],
        pb_image["path"],
        skymodel_pb["path"] if sector["save_source_list"] else "none",
        skymodel_nonpb["path"] if sector["save_source_list"] else "none",
        image_name,
        str(sector["vertices_file"]),
        list(sector["obs_original_paths"]),
        float(sector["threshisl"]),
        float(sector["threshpix"]),
        bool(sector["filter_by_mask"]),
        str(sector["source_finder"]),
        int(sector["max_threads"]),
        bright_true_sky_skymodel=sector.get("bright_skymodel_pb"),
    )
    _run_shell(
        filter_command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls
    )

    filtered_true_sky = _require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.true_sky.txt"),
        "Filtered true-sky skymodel",
    )
    filtered_apparent_sky = _require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.apparent_sky.txt"),
        "Filtered apparent-sky skymodel",
    )
    diagnostics = _require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.image_diagnostics.json"),
        "Image diagnostics",
    )
    flat_noise_rms = _require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.flat_noise_rms.fits"),
        "Flat-noise RMS image",
    )
    true_sky_rms = _require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.true_sky_rms.fits"),
        "True-sky RMS image",
    )
    source_catalog = _require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.source_catalog.fits"),
        "Source catalog",
    )
    source_filtering_mask_path = os.path.join(
        pipeline_working_dir, f"{os.path.basename(pb_image['path'])}.mask.fits"
    )
    source_filtering_mask = (
        file_record(source_filtering_mask_path)
        if os.path.isfile(source_filtering_mask_path)
        else None
    )

    skymodel_image = None
    if sector["save_filtered_model_image"]:
        skymodel_image_command = build_make_skymodel_image_command(
            filtered_apparent_sky["path"],
            pb_image["path"],
            str(sector["filtered_model_image_filename"]),
        )
        _run_shell(
            skymodel_image_command,
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )
        skymodel_image = _require_file(
            str(sector["filtered_model_image_path"]), "Filtered skymodel image"
        )

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
    _run_shell(
        diagnostics_command, pipeline_working_dir, config, shell_operation_cls=shell_operation_cls
    )
    diagnostics = _require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.image_diagnostics.json"),
        "Image diagnostics",
    )
    offsets_path = os.path.join(pipeline_working_dir, f"{image_name}.astrometry_offsets.json")
    offsets = file_record(offsets_path) if os.path.isfile(offsets_path) else None
    diagnostic_plots = _file_records_for_patterns(
        [os.path.join(pipeline_working_dir, f"{image_name}*.pdf")]
    )
    publish_plot_file_records([diagnostics], pipeline_working_dir)
    publish_plot_file_records(diagnostic_plots, pipeline_working_dir)

    output_sector_images = sector_images
    output_extra_images = extra_images
    if sector["compress_images"]:
        output_sector_images, output_extra_images = _compress_image_records(
            image_name,
            sector_images,
            extra_images,
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )

    normalization_source_catalog = None
    normalize_h5parm = None
    if sector["normalize_flux_scale"]:
        normalization_source_catalog, normalize_h5parm = _make_normalization_records(
            image_cubes[0],
            image_cube_beams[0],
            image_cube_frequencies[0],
            concat_record,
            sector,
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )

    result = {
        "filtered_skymodel_true_sky": filtered_true_sky,
        "filtered_skymodel_apparent_sky": filtered_apparent_sky,
        "pybdsf_catalog": source_catalog,
        "sector_diagnostics": diagnostics,
        "sector_offsets": offsets,
        "sector_diagnostic_plots": diagnostic_plots,
        "visibilities": prepared_records,
        "sector_I_images": output_sector_images,
        "sector_extra_images": output_extra_images,
        "source_filtering_mask": source_filtering_mask,
        "sector_skymodels": [skymodel_nonpb, skymodel_pb] if sector["save_source_list"] else None,
    }
    if region_record is not None:
        result["sector_region_file"] = region_record
    if skymodel_image is not None:
        result["sector_skymodel_image_fits"] = skymodel_image
    if image_cubes:
        result["sector_image_cubes"] = image_cubes
        result["sector_image_cube_beams"] = image_cube_beams
        result["sector_image_cube_frequencies"] = image_cube_frequencies
    if normalize_h5parm is not None:
        result["sector_source_catalog"] = normalization_source_catalog
        result["sector_normalize_h5parm"] = normalize_h5parm
    fits_records = (
        output_sector_images
        + output_extra_images
        + [flat_noise_rms, true_sky_rms, source_catalog]
        + ([source_filtering_mask] if source_filtering_mask is not None else [])
        + ([skymodel_image] if skymodel_image is not None else [])
        + image_cubes
    )
    publish_fits_image_artifacts(fits_records, pipeline_working_dir)
    return result


@task(name="image_sector")
def image_sector_task(
    sector: ImageSectorPayload,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prefect task wrapper for one imaging sector."""
    with publish_python_logs_to_prefect():
        return run_image_sector(
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
        run_image_sector(
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
