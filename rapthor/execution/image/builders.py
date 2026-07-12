"""Payload builders for image execution."""

import os
from typing import Mapping, Optional

from rapthor.execution.image.payloads import (
    ImageCubeSpecPayload,
    ImagePayload,
    ImagePrepareTaskPayload,
    ImageSectorPayload,
)
from rapthor.execution.payloads import assert_serializable_payload, validate_basename
from rapthor.lib.records import (
    directory_record_path,
    file_record_path,
    optional_file_record_path,
)


def image_payload_from_inputs(
    input_parms: Mapping[str, object],
    pipeline_working_dir: object,
    *,
    apply_screens: bool = False,
    use_facets: bool = False,
    compress_images: bool = False,
    make_image_cube: bool = False,
    make_residual_visibilities: bool = False,
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
    make_residual_visibilities = bool(
        make_residual_visibilities or input_parms.get("make_residual_visibilities", False)
    )
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
        "parallel_gridding_tasks",
    ]
    if make_residual_visibilities:
        per_sector_keys.append("residual_filename")
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
    prepare_data_h5parm = optional_file_record_path(input_parms.get("prepare_data_h5parm"))
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
        for key in ["soltabs", "scalar_visibilities", "diagonal_visibilities", "shared_facet_rw"]:
            if key not in input_parms:
                raise ValueError(f"{key} is required when use_facets=True")
    fulljones_h5parm = optional_file_record_path(input_parms.get("fulljones_h5parm"))
    input_normalize_h5parm = optional_file_record_path(input_parms.get("input_normalize_h5parm"))
    photometry_skymodel = optional_file_record_path(input_parms.get("photometry_skymodel"))
    astrometry_skymodel = optional_file_record_path(input_parms.get("astrometry_skymodel"))
    obs_original_filename = input_parms.get("obs_original_filename", input_parms["obs_filename"])
    if not isinstance(obs_original_filename, list) or len(obs_original_filename) != sector_count:
        raise ValueError("obs_original_filename must be a list with one value per sector")

    sectors: list[ImageSectorPayload] = []
    for sector_index in range(sector_count):
        obs_records = input_parms["obs_filename"][sector_index]
        obs_original_records = obs_original_filename[sector_index]
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
        if not all(isinstance(value, list) for value in [obs_original_records, *obs_inputs]):
            raise ValueError(f"sector {sector_index} observation inputs must be lists")
        obs_count = len(obs_records)
        if any(len(value) != obs_count for value in obs_inputs):
            raise ValueError(f"sector {sector_index} observation inputs must have the same length")
        if len(obs_original_records) != obs_count:
            raise ValueError(
                f"sector {sector_index} original observation inputs must match obs_filename"
            )

        prepare_tasks: list[ImagePrepareTaskPayload] = []
        for obs_index in range(obs_count):
            msout = validate_basename(
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

        image_name = validate_basename(image_names[sector_index], f"image_name[{sector_index}]")
        concat_filename = validate_basename(
            input_parms["concat_filename"][sector_index], f"concat_filename[{sector_index}]"
        )
        residual_filename = None
        if make_residual_visibilities:
            residual_filename = validate_basename(
                input_parms["residual_filename"][sector_index],
                f"residual_filename[{sector_index}]",
            )
        mask_filename = validate_basename(
            input_parms["mask_filename"][sector_index], f"mask_filename[{sector_index}]"
        )
        facet_region_filename = None
        if use_facets:
            facet_region_filename = validate_basename(
                input_parms["facet_region_file"][sector_index],
                f"facet_region_file[{sector_index}]",
            )
        filtered_model_image_filename = None
        if save_filtered_model_image:
            filtered_model_image_filename = validate_basename(
                input_parms["filtered_model_image_name"][sector_index],
                f"filtered_model_image_name[{sector_index}]",
            )
        image_i_cube_filename = None
        image_cube_specs: list[ImageCubeSpecPayload] = []
        if make_image_cube:
            image_i_cube_filename = validate_basename(
                input_parms["image_I_cube_name"][sector_index],
                f"image_I_cube_name[{sector_index}]",
            )
            for stokes in pol.upper():
                key = f"image_{stokes}_cube_name"
                if key not in input_parms:
                    continue
                image_cube_filename = validate_basename(
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
            output_source_catalog_filename = validate_basename(
                input_parms["output_source_catalog"][sector_index],
                f"output_source_catalog[{sector_index}]",
            )
            output_normalize_h5parm_filename = validate_basename(
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
                "make_residual_visibilities": make_residual_visibilities,
                "normalize_flux_scale": bool(normalize_flux_scale),
                "peel_bright_sources": peel_bright_sources,
                "save_filtered_model_image": save_filtered_model_image,
                "bright_skymodel_pb": bright_skymodel_pb,
                "data_colname": str(input_parms["data_colname"]),
                "prepare_tasks": prepare_tasks,
                "concat_filename": concat_filename,
                "concat_path": os.path.join(pipeline_dir, concat_filename),
                "residual_filename": residual_filename,
                "residual_path": (
                    None
                    if residual_filename is None
                    else os.path.join(pipeline_dir, residual_filename)
                ),
                "previous_mask_filename": optional_file_record_path(
                    input_parms["previous_mask_filename"][sector_index]
                ),
                "mask_filename": mask_filename,
                "mask_path": os.path.join(pipeline_dir, mask_filename),
                "timebase": input_parms["image_timebase"][sector_index],
                "phasecenter": str(input_parms["phasecenter"][sector_index]),
                "h5parm": h5parm,
                "prepare_data_h5parm": prepare_data_h5parm,
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
                "normalization_skymodels": _optional_file_record_paths(
                    input_parms.get("normalization_skymodels"),
                    "normalization_skymodels",
                ),
                "normalization_reference_frequencies": input_parms.get(
                    "normalization_reference_frequencies"
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
                "parallel_gridding_tasks": int(
                    input_parms["parallel_gridding_tasks"][sector_index]
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
                "filter_skymodel_ncores": int(
                    input_parms.get("filter_skymodel_ncores", input_parms["max_threads"])
                ),
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
                "obs_original_paths": [
                    directory_record_path(record) for record in obs_original_records
                ],
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


def _pol_token(pol: object) -> str:
    if isinstance(pol, str):
        return pol
    if isinstance(pol, list):
        return "".join(str(value) for value in pol)
    raise ValueError("pol must be a string or list")


def _is_stokes_i(pol: str) -> bool:
    return pol.upper() == "I"


def _optional_file_record_paths(records: object, label: str) -> Optional[list[str]]:
    if not records:
        return None
    if not isinstance(records, list):
        raise ValueError(f"{label} must be a list of File records")
    return [file_record_path(record) for record in records]
