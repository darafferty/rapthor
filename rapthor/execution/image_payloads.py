"""Image execution payload builders and validators."""

import os
from typing import Mapping

from rapthor.execution.payloads import (
    ImageCubeSpecPayload,
    ImagePayload,
    ImagePrepareTaskPayload,
    ImageSectorPayload,
    assert_serializable_payload,
)
from rapthor.lib.records import (
    directory_record_path,
    file_record_path,
    optional_file_record_path,
)


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


def validate_image_payload(payload: Mapping[str, object]) -> ImagePayload:
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
