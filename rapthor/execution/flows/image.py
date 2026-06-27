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
from rapthor.execution.outputs import (
    directory_record,
    directory_record_path,
    file_record,
    file_record_path,
    optional_file_record_path,
    validate_output_record,
)
from rapthor.execution.payloads import (
    ImageCubeSpecPayload,
    ImagePayload,
    ImagePrepareTaskPayload,
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


def _validate_basename(filename: object, name: str) -> str:
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{name} must be a non-empty string")
    if os.path.isabs(filename) or os.path.basename(filename) != filename:
        raise ValueError(f"{name} must be a basename")
    return filename


def _pol_token(pol: object) -> str:
    if isinstance(pol, str):
        return pol
    if isinstance(pol, list):
        return "".join(str(value) for value in pol)
    raise ValueError("pol must be a string or list")


def _is_stokes_i(pol: str) -> bool:
    return pol.upper() == "I"


def image_payload_from_inputs(
    input_parms: Mapping[str, object],
    pipeline_working_dir: object,
    *,
    apply_screens: bool = False,
    use_facets: bool = False,
    compress_images: bool = False,
    make_image_cube: bool = False,
    normalize_flux_scale: bool = False,
    use_mpi: bool = False,
) -> ImagePayload:
    """Create a serializable Image flow payload."""
    if apply_screens and use_facets:
        raise ValueError("apply_screens and use_facets cannot both be enabled")
    if normalize_flux_scale and not make_image_cube:
        raise ValueError("normalize_flux_scale requires make_image_cube=True")

    pol = _pol_token(input_parms["pol"])
    peel_bright_sources = bool(input_parms.get("peel_bright_sources", False))
    bright_skymodel_pb = optional_file_record_path(input_parms.get("bright_skymodel_pb"))
    if peel_bright_sources and bright_skymodel_pb is None:
        raise ValueError("bright_skymodel_pb must be a File record when peel_bright_sources=True")

    pipeline_dir = str(pipeline_working_dir)
    image_names = input_parms.get("image_name", [])
    if not isinstance(image_names, list):
        raise ValueError("image_name must be a list")
    sector_count = len(image_names)
    save_filtered_model_image = bool(input_parms.get("save_filtered_model_image"))
    per_sector_keys = [
        "obs_filename",
        "prepare_filename",
        "concat_filename",
        "previous_mask_filename",
        "mask_filename",
        "starttime",
        "ntimes",
        "image_freqstep",
        "image_timestep",
        "image_maxinterval",
        "image_timebase",
        "phasecenter",
        "channels_out",
        "deconvolution_channels",
        "fit_spectral_pol",
        "ra",
        "dec",
        "wsclean_imsize",
        "vertices_file",
        "region_file",
        "wsclean_niter",
        "wsclean_nmiter",
        "robust",
        "cellsize_deg",
        "min_uv_lambda",
        "max_uv_lambda",
        "mgain",
        "taper_arcsec",
        "local_rms_strength",
        "local_rms_window",
        "local_rms_method",
        "auto_mask",
        "auto_mask_nmiter",
        "idg_mode",
        "wsclean_mem",
        "threshisl",
        "threshpix",
        "do_multiscale",
        "dd_psf_grid",
    ]
    if use_facets:
        per_sector_keys.extend(
            [
                "ra_mid",
                "dec_mid",
                "width_ra",
                "width_dec",
                "facet_region_file",
            ]
        )
    if save_filtered_model_image:
        per_sector_keys.append("filtered_model_image_name")
    if make_image_cube:
        per_sector_keys.append("image_I_cube_name")
    if normalize_flux_scale:
        per_sector_keys.extend(["output_source_catalog", "output_normalize_h5parm"])
    if use_mpi:
        per_sector_keys.extend(["mpi_nnodes", "mpi_cpus_per_task"])
    for key in per_sector_keys:
        value = input_parms.get(key, [])
        if not isinstance(value, list) or len(value) != sector_count:
            raise ValueError(f"{key} must be a list with one value per sector")

    h5parm = optional_file_record_path(input_parms.get("h5parm"))
    if (apply_screens or use_facets) and h5parm is None:
        raise ValueError("h5parm must be a File record for screen or facet imaging")
    interval = None
    if apply_screens:
        interval_value = input_parms.get("interval")
        if (
            not isinstance(interval_value, list)
            or len(interval_value) != 2
            or not all(isinstance(value, int) for value in interval_value)
        ):
            raise ValueError("interval must be a two-element integer list when apply_screens=True")
        interval = [int(value) for value in interval_value]
    facet_skymodel = None
    if use_facets:
        facet_skymodel = file_record_path(input_parms.get("skymodel"))
        for key in [
            "soltabs",
            "parallel_gridding_threads",
            "scalar_visibilities",
            "diagonal_visibilities",
            "shared_facet_rw",
        ]:
            if key not in input_parms:
                raise ValueError(f"{key} is required when use_facets=True")
    fulljones_h5parm = optional_file_record_path(input_parms.get("fulljones_h5parm"))
    input_normalize_h5parm = optional_file_record_path(input_parms.get("input_normalize_h5parm"))
    photometry_skymodel = optional_file_record_path(input_parms.get("photometry_skymodel"))
    astrometry_skymodel = optional_file_record_path(input_parms.get("astrometry_skymodel"))

    sectors: list[ImageSectorPayload] = []
    for sector_index in range(sector_count):
        obs_records = input_parms["obs_filename"][sector_index]
        prepare_filenames = input_parms["prepare_filename"][sector_index]
        starttimes = input_parms["starttime"][sector_index]
        ntimes = input_parms["ntimes"][sector_index]
        freqsteps = input_parms["image_freqstep"][sector_index]
        timesteps = input_parms["image_timestep"][sector_index]
        maxintervals = input_parms["image_maxinterval"][sector_index]
        obs_inputs = [
            obs_records,
            prepare_filenames,
            starttimes,
            ntimes,
            freqsteps,
            timesteps,
            maxintervals,
        ]
        if not all(isinstance(value, list) for value in obs_inputs):
            raise ValueError(f"sector {sector_index} observation inputs must be lists")
        obs_count = len(obs_records)
        if any(len(value) != obs_count for value in obs_inputs):
            raise ValueError(f"sector {sector_index} observation inputs must have the same length")

        prepare_tasks: list[ImagePrepareTaskPayload] = []
        for obs_index in range(obs_count):
            msout = _validate_basename(
                prepare_filenames[obs_index], f"prepare_filename[{sector_index}][{obs_index}]"
            )
            prepare_tasks.append(
                {
                    "msin": directory_record_path(obs_records[obs_index]),
                    "msout": msout,
                    "msout_path": os.path.join(pipeline_dir, msout),
                    "starttime": str(starttimes[obs_index]),
                    "ntimes": int(ntimes[obs_index]),
                    "freqstep": int(freqsteps[obs_index]),
                    "timestep": int(timesteps[obs_index]),
                    "maxinterval": (
                        None if maxintervals[obs_index] is None else int(maxintervals[obs_index])
                    ),
                }
            )

        image_name = _validate_basename(image_names[sector_index], f"image_name[{sector_index}]")
        concat_filename = _validate_basename(
            input_parms["concat_filename"][sector_index], f"concat_filename[{sector_index}]"
        )
        mask_filename = _validate_basename(
            input_parms["mask_filename"][sector_index], f"mask_filename[{sector_index}]"
        )
        facet_region_filename = None
        if use_facets:
            facet_region_filename = _validate_basename(
                input_parms["facet_region_file"][sector_index],
                f"facet_region_file[{sector_index}]",
            )
        filtered_model_image_filename = None
        if save_filtered_model_image:
            filtered_model_image_filename = _validate_basename(
                input_parms["filtered_model_image_name"][sector_index],
                f"filtered_model_image_name[{sector_index}]",
            )
        image_i_cube_filename = None
        image_cube_specs: list[ImageCubeSpecPayload] = []
        if make_image_cube:
            image_i_cube_filename = _validate_basename(
                input_parms["image_I_cube_name"][sector_index],
                f"image_I_cube_name[{sector_index}]",
            )
            for stokes in pol.upper():
                key = f"image_{stokes}_cube_name"
                if key not in input_parms:
                    continue
                image_cube_filename = _validate_basename(
                    input_parms[key][sector_index],
                    f"{key}[{sector_index}]",
                )
                image_cube_specs.append(
                    {
                        "pol": stokes,
                        "filename": image_cube_filename,
                        "path": os.path.join(pipeline_dir, image_cube_filename),
                    }
                )
            if not image_cube_specs:
                image_cube_specs.append(
                    {
                        "pol": "I",
                        "filename": image_i_cube_filename,
                        "path": os.path.join(pipeline_dir, image_i_cube_filename),
                    }
                )
        output_source_catalog_filename = None
        output_normalize_h5parm_filename = None
        if normalize_flux_scale:
            output_source_catalog_filename = _validate_basename(
                input_parms["output_source_catalog"][sector_index],
                f"output_source_catalog[{sector_index}]",
            )
            output_normalize_h5parm_filename = _validate_basename(
                input_parms["output_normalize_h5parm"][sector_index],
                f"output_normalize_h5parm[{sector_index}]",
            )
        sectors.append(
            {
                "image_name": image_name,
                "apply_screens": apply_screens,
                "use_facets": use_facets,
                "use_mpi": bool(use_mpi),
                "compress_images": bool(compress_images),
                "make_image_cube": bool(make_image_cube),
                "normalize_flux_scale": bool(normalize_flux_scale),
                "peel_bright_sources": peel_bright_sources,
                "save_filtered_model_image": save_filtered_model_image,
                "bright_skymodel_pb": bright_skymodel_pb,
                "data_colname": str(input_parms["data_colname"]),
                "prepare_tasks": prepare_tasks,
                "concat_filename": concat_filename,
                "concat_path": os.path.join(pipeline_dir, concat_filename),
                "previous_mask_filename": optional_file_record_path(
                    input_parms["previous_mask_filename"][sector_index]
                ),
                "mask_filename": mask_filename,
                "mask_path": os.path.join(pipeline_dir, mask_filename),
                "timebase": input_parms["image_timebase"][sector_index],
                "phasecenter": str(input_parms["phasecenter"][sector_index]),
                "h5parm": h5parm,
                "fulljones_h5parm": fulljones_h5parm,
                "input_normalize_h5parm": input_normalize_h5parm,
                "prepare_data_steps": str(input_parms["prepare_data_steps"]),
                "prepare_data_applycal_steps": input_parms.get("prepare_data_applycal_steps"),
                "central_patch_name": (
                    input_parms.get("central_patch_name", [None] * sector_count)[sector_index]
                    if isinstance(input_parms.get("central_patch_name", []), list)
                    else None
                ),
                "channels_out": int(input_parms["channels_out"][sector_index]),
                "deconvolution_channels": int(input_parms["deconvolution_channels"][sector_index]),
                "fit_spectral_pol": int(input_parms["fit_spectral_pol"][sector_index]),
                "ra": float(input_parms["ra"][sector_index]),
                "dec": float(input_parms["dec"][sector_index]),
                "wsclean_imsize": [
                    int(value) for value in input_parms["wsclean_imsize"][sector_index]
                ],
                "vertices_file": file_record_path(input_parms["vertices_file"][sector_index]),
                "region_file": optional_file_record_path(input_parms["region_file"][sector_index]),
                "facet_skymodel": facet_skymodel,
                "facet_region_filename": facet_region_filename,
                "facet_region_path": (
                    None
                    if facet_region_filename is None
                    else os.path.join(pipeline_dir, facet_region_filename)
                ),
                "filtered_model_image_filename": filtered_model_image_filename,
                "filtered_model_image_path": (
                    None
                    if filtered_model_image_filename is None
                    else os.path.join(pipeline_dir, filtered_model_image_filename)
                ),
                "image_I_cube_filename": image_i_cube_filename,
                "image_I_cube_path": (
                    None
                    if image_i_cube_filename is None
                    else os.path.join(pipeline_dir, image_i_cube_filename)
                ),
                "image_cube_specs": image_cube_specs,
                "output_source_catalog_filename": output_source_catalog_filename,
                "output_source_catalog_path": (
                    None
                    if output_source_catalog_filename is None
                    else os.path.join(pipeline_dir, output_source_catalog_filename)
                ),
                "output_normalize_h5parm_filename": output_normalize_h5parm_filename,
                "output_normalize_h5parm_path": (
                    None
                    if output_normalize_h5parm_filename is None
                    else os.path.join(pipeline_dir, output_normalize_h5parm_filename)
                ),
                "ra_mid": (None if not use_facets else float(input_parms["ra_mid"][sector_index])),
                "dec_mid": (
                    None if not use_facets else float(input_parms["dec_mid"][sector_index])
                ),
                "width_ra": (
                    None if not use_facets else float(input_parms["width_ra"][sector_index])
                ),
                "width_dec": (
                    None if not use_facets else float(input_parms["width_dec"][sector_index])
                ),
                "wsclean_niter": int(input_parms["wsclean_niter"][sector_index]),
                "wsclean_nmiter": int(input_parms["wsclean_nmiter"][sector_index]),
                "skip_final_iteration": bool(input_parms["skip_final_iteration"]),
                "robust": float(input_parms["robust"][sector_index]),
                "cellsize_deg": float(input_parms["cellsize_deg"][sector_index]),
                "min_uv_lambda": float(input_parms["min_uv_lambda"][sector_index]),
                "max_uv_lambda": float(input_parms["max_uv_lambda"][sector_index]),
                "mgain": float(input_parms["mgain"][sector_index]),
                "taper_arcsec": float(input_parms["taper_arcsec"][sector_index]),
                "local_rms_strength": float(input_parms["local_rms_strength"][sector_index]),
                "local_rms_window": float(input_parms["local_rms_window"][sector_index]),
                "local_rms_method": str(input_parms["local_rms_method"][sector_index]),
                "auto_mask": float(input_parms["auto_mask"][sector_index]),
                "auto_mask_nmiter": int(input_parms["auto_mask_nmiter"][sector_index]),
                "idg_mode": str(input_parms["idg_mode"][sector_index]),
                "wsclean_mem": float(input_parms["wsclean_mem"][sector_index]),
                "threshisl": float(input_parms["threshisl"][sector_index]),
                "threshpix": float(input_parms["threshpix"][sector_index]),
                "do_multiscale": bool(input_parms["do_multiscale"][sector_index]),
                "dd_psf_grid": [int(value) for value in input_parms["dd_psf_grid"][sector_index]],
                "interval": interval,
                "soltabs": None if not use_facets else str(input_parms["soltabs"]),
                "parallel_gridding_threads": (
                    None if not use_facets else int(input_parms["parallel_gridding_threads"])
                ),
                "scalar_visibilities": (
                    None if not use_facets else bool(input_parms["scalar_visibilities"])
                ),
                "diagonal_visibilities": (
                    None if not use_facets else bool(input_parms["diagonal_visibilities"])
                ),
                "shared_facet_reads": (
                    None if not use_facets else bool(input_parms["shared_facet_rw"])
                ),
                "shared_facet_writes": (
                    None if not use_facets else bool(input_parms["shared_facet_rw"])
                ),
                "pol": pol,
                "save_source_list": bool(input_parms["save_source_list"]),
                "link_polarizations": input_parms["link_polarizations"],
                "join_polarizations": bool(input_parms["join_polarizations"]),
                "filter_by_mask": bool(input_parms["filter_by_mask"]),
                "source_finder": str(input_parms["source_finder"]),
                "apply_time_frequency_smearing": bool(input_parms["apply_time_frequency_smearing"]),
                "max_threads": int(input_parms["max_threads"]),
                "deconvolution_threads": int(input_parms["deconvolution_threads"]),
                "mpi_nnodes": (
                    None if not use_mpi else int(input_parms["mpi_nnodes"][sector_index])
                ),
                "mpi_cpus_per_task": (
                    None if not use_mpi else int(input_parms["mpi_cpus_per_task"][sector_index])
                ),
                "allow_internet_access": bool(input_parms["allow_internet_access"]),
                "photometry_skymodel": photometry_skymodel,
                "astrometry_skymodel": astrometry_skymodel,
                "obs_original_paths": [directory_record_path(record) for record in obs_records],
                "obs_starttime": [str(value) for value in starttimes],
                "obs_ntimes": [int(value) for value in ntimes],
            }
        )

    stokes_mode = "stokes_i" if _is_stokes_i(pol) else "full_stokes"
    if apply_screens:
        mode = f"screens_{stokes_mode}"
    elif use_facets:
        mode = f"facet_{stokes_mode}"
    else:
        mode = f"no_dde_{stokes_mode}"

    payload: ImagePayload = {
        "mode": mode,
        "use_mpi": bool(use_mpi),
        "pipeline_working_dir": pipeline_dir,
        "sectors": sectors,
    }
    assert_serializable_payload(payload)
    return payload


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


def _validate_int_list(values: object, name: str) -> list[int]:
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise ValueError(f"{name} must be a list of integers")
    return list(values)


def _validate_str_list(values: object, name: str) -> list[str]:
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise ValueError(f"{name} must be a list of strings")
    return list(values)


def _validate_prepare_task(
    prepare_task: Mapping[str, object],
    sector_index: int,
    task_index: int,
) -> ImagePrepareTaskPayload:
    maxinterval = prepare_task.get("maxinterval")
    return {
        "msin": str(prepare_task["msin"]),
        "msout": _validate_basename(
            prepare_task["msout"],
            f"sectors[{sector_index}].prepare_tasks[{task_index}].msout",
        ),
        "msout_path": str(prepare_task["msout_path"]),
        "starttime": str(prepare_task["starttime"]),
        "ntimes": int(prepare_task["ntimes"]),
        "freqstep": int(prepare_task["freqstep"]),
        "timestep": int(prepare_task["timestep"]),
        "maxinterval": None if maxinterval is None else int(maxinterval),
    }


def _validate_image_cube_spec(
    spec: Mapping[str, object],
    sector_index: int,
    spec_index: int,
) -> ImageCubeSpecPayload:
    return {
        "pol": str(spec["pol"]),
        "filename": _validate_basename(
            spec["filename"],
            f"sectors[{sector_index}].image_cube_specs[{spec_index}].filename",
        ),
        "path": str(spec["path"]),
    }


def _validate_image_sector(sector: Mapping[str, object], index: int) -> ImageSectorPayload:
    raw_prepare_tasks = sector.get("prepare_tasks", [])
    if not isinstance(raw_prepare_tasks, list):
        raise ValueError(f"sectors[{index}].prepare_tasks must be a list")
    prepare_tasks = []
    for task_index, prepare_task in enumerate(raw_prepare_tasks):
        if not isinstance(prepare_task, Mapping):
            raise ValueError(f"sectors[{index}].prepare_tasks[{task_index}] must be a mapping")
        prepare_tasks.append(_validate_prepare_task(prepare_task, index, task_index))

    raw_image_cube_specs = sector.get("image_cube_specs", [])
    if not isinstance(raw_image_cube_specs, list):
        raise ValueError(f"sectors[{index}].image_cube_specs must be a list")
    image_cube_specs = []
    for spec_index, spec in enumerate(raw_image_cube_specs):
        if not isinstance(spec, Mapping):
            raise ValueError(f"sectors[{index}].image_cube_specs[{spec_index}] must be a mapping")
        image_cube_specs.append(_validate_image_cube_spec(spec, index, spec_index))

    validated_sector = dict(sector)
    validated_sector["prepare_tasks"] = prepare_tasks
    validated_sector["image_cube_specs"] = image_cube_specs
    validated_sector["wsclean_imsize"] = _validate_int_list(
        sector.get("wsclean_imsize"),
        f"sectors[{index}].wsclean_imsize",
    )
    validated_sector["dd_psf_grid"] = _validate_int_list(
        sector.get("dd_psf_grid"),
        f"sectors[{index}].dd_psf_grid",
    )
    validated_sector["obs_original_paths"] = _validate_str_list(
        sector.get("obs_original_paths"),
        f"sectors[{index}].obs_original_paths",
    )
    validated_sector["obs_starttime"] = _validate_str_list(
        sector.get("obs_starttime"),
        f"sectors[{index}].obs_starttime",
    )
    validated_sector["obs_ntimes"] = _validate_int_list(
        sector.get("obs_ntimes"),
        f"sectors[{index}].obs_ntimes",
    )
    return validated_sector


def _validate_image_payload(payload: Mapping[str, object]) -> ImagePayload:
    supported_modes = {
        "facet_full_stokes",
        "facet_stokes_i",
        "no_dde_full_stokes",
        "no_dde_stokes_i",
        "screens_full_stokes",
        "screens_stokes_i",
    }
    mode = str(payload["mode"])
    if mode not in supported_modes:
        raise ValueError("Only no-DDE, facet, and screen image payloads are supported")
    pipeline_working_dir = str(payload["pipeline_working_dir"])
    raw_sectors = payload.get("sectors", [])
    if not isinstance(raw_sectors, list):
        raise ValueError("sectors must be a list")
    sectors = []
    for index, sector in enumerate(raw_sectors):
        if not isinstance(sector, Mapping):
            raise ValueError(f"sectors[{index}] must be a mapping")
        sectors.append(_validate_image_sector(sector, index))
    return {
        "mode": mode,
        "use_mpi": bool(payload.get("use_mpi", False)),
        "pipeline_working_dir": pipeline_working_dir,
        "sectors": sectors,
    }


def run_image_flow(
    payload: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run imaging commands and return finalizer-compatible outputs."""
    assert_serializable_payload(payload)
    config = execution_config or ExecutionConfig(task_runner="sync")
    payload = _validate_image_payload(payload)
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
    payload = _validate_image_payload(payload)
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
