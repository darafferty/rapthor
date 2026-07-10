"""Execution helpers for one image sector."""

from typing import Mapping, Optional

from rapthor.execution.artifacts import (
    publish_fits_image_artifacts,
    publish_fits_postage_stamp_artifacts,
)
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.image.astrometry import make_astrometry_corrected_image_record
from rapthor.execution.image.contracts import ImageSectorPayload
from rapthor.execution.image.diagnostics import run_image_diagnostics
from rapthor.execution.image.outputs import (
    compress_image_records,
    filter_skymodel_products,
    make_filtered_model_image,
    make_image_cube_records,
    make_normalization_records,
    mfs_extra_image_patterns,
    source_list_records,
)
from rapthor.execution.image.preparation import (
    ensure_facet_region,
    ensure_imaging_mask,
    prepare_and_concatenate_visibilities,
)
from rapthor.execution.image.residual_visibilities import make_residual_visibility_record
from rapthor.execution.image.wsclean import (
    check_wsclean_beams,
    restore_bright_source_images,
    run_or_reuse_wsclean_images,
)
from rapthor.execution.outputs import file_records_for_patterns


def prepare_image_sector(
    sector: ImageSectorPayload,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run data preparation and WSClean for one imaging sector."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    image_name = str(sector["image_name"])

    prepared_records, concat_record = prepare_and_concatenate_visibilities(
        sector,
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )
    mask_record = ensure_imaging_mask(sector)
    region_record = ensure_facet_region(sector)

    nonpb_image, pb_image, wsclean_ran = run_or_reuse_wsclean_images(
        sector,
        concat_record,
        mask_record,
        region_record,
        image_name,
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )
    extra_images = file_records_for_patterns(
        mfs_extra_image_patterns(image_name, pipeline_working_dir)
    )
    nonpb_image, pb_image = restore_bright_source_images(
        sector,
        nonpb_image,
        pb_image,
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )
    residual_visibilities = make_residual_visibility_record(
        sector,
        concat_record,
        pipeline_working_dir,
        config,
        force=wsclean_ran,
        shell_operation_cls=shell_operation_cls,
    )
    sector_images = [nonpb_image, pb_image]

    skymodel_nonpb, skymodel_pb = source_list_records(
        sector,
        image_name,
        pipeline_working_dir,
    )
    if wsclean_ran:
        check_wsclean_beams(
            (pb_image, nonpb_image),
            sector,
        )

    return {
        "prepared_records": prepared_records,
        "concat_record": concat_record,
        "mask_record": mask_record,
        "region_record": region_record,
        "nonpb_image": nonpb_image,
        "pb_image": pb_image,
        "wsclean_ran": wsclean_ran,
        "extra_images": extra_images,
        "residual_visibilities": residual_visibilities,
        "sector_images": sector_images,
        "skymodel_nonpb": skymodel_nonpb,
        "skymodel_pb": skymodel_pb,
    }


def filter_image_sector_skymodel(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Filter one sector skymodel and return diagnostics inputs."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    image_name = str(sector["image_name"])
    (
        filtered_true_sky,
        filtered_apparent_sky,
        diagnostics,
        flat_noise_rms,
        true_sky_rms,
        source_catalog,
        source_filtering_mask,
    ) = filter_skymodel_products(
        sector,
        image_name,
        prepared["nonpb_image"],
        prepared["pb_image"],
        prepared["skymodel_nonpb"],
        prepared["skymodel_pb"],
        pipeline_working_dir,
        execution_config=config,
        shell_operation_cls=shell_operation_cls,
    )
    return {
        "filtered_true_sky": filtered_true_sky,
        "filtered_apparent_sky": filtered_apparent_sky,
        "diagnostics": diagnostics,
        "flat_noise_rms": flat_noise_rms,
        "true_sky_rms": true_sky_rms,
        "source_catalog": source_catalog,
        "source_filtering_mask": source_filtering_mask,
    }


def calculate_image_sector_diagnostics(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    filtered: Mapping[str, object],
    pipeline_working_dir: str,
) -> dict:
    """Calculate image diagnostics for one sector and return output records."""
    image_name = str(sector["image_name"])
    diagnostics, offsets, diagnostic_plots = run_image_diagnostics(
        sector,
        image_name,
        prepared["nonpb_image"],
        prepared["pb_image"],
        filtered["flat_noise_rms"],
        filtered["true_sky_rms"],
        filtered["source_catalog"],
        filtered["diagnostics"],
        prepared["region_record"],
        pipeline_working_dir,
    )
    return {
        "diagnostics": diagnostics,
        "offsets": offsets,
        "diagnostic_plots": diagnostic_plots,
    }


def make_image_sector_cubes(
    sector: ImageSectorPayload,
    pipeline_working_dir: str,
) -> dict:
    """Build requested image cubes for one sector."""
    if not sector["make_image_cube"]:
        return {
            "image_cubes": [],
            "image_cube_beams": [],
            "image_cube_frequencies": [],
        }

    image_cubes, image_cube_beams, image_cube_frequencies = make_image_cube_records(
        str(sector["image_name"]),
        list(sector["image_cube_specs"]),
        pipeline_working_dir,
    )
    return {
        "image_cubes": image_cubes,
        "image_cube_beams": image_cube_beams,
        "image_cube_frequencies": image_cube_frequencies,
    }


def normalize_image_sector_flux_scale(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    image_cube_result: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Build flux-scale normalization products for one sector when requested."""
    if not sector["normalize_flux_scale"]:
        return {"source_catalog": None, "normalize_h5parm": None}

    image_cubes = list(image_cube_result["image_cubes"])
    image_cube_beams = list(image_cube_result["image_cube_beams"])
    image_cube_frequencies = list(image_cube_result["image_cube_frequencies"])
    if not image_cubes:
        raise ValueError("normalize_flux_scale requires image cube outputs")

    config = execution_config or ExecutionConfig(task_runner="sync")
    source_catalog, normalize_h5parm = make_normalization_records(
        image_cubes[0],
        image_cube_beams[0],
        image_cube_frequencies[0],
        prepared["concat_record"],
        sector,
        config,
        shell_operation_cls=shell_operation_cls,
    )
    return {
        "source_catalog": source_catalog,
        "normalize_h5parm": normalize_h5parm,
    }


def finalize_image_sector(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    filtered: Mapping[str, object],
    diagnostics_result: Mapping[str, object],
    pipeline_working_dir: str,
    image_cube_result: Optional[Mapping[str, object]] = None,
    normalization_result: Optional[Mapping[str, object]] = None,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Build post-WSClean sector products and assemble the output record."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    image_name = str(sector["image_name"])
    prepared_records = list(prepared["prepared_records"])
    region_record = prepared["region_record"]
    pb_image = prepared["pb_image"]
    extra_images = list(prepared["extra_images"])
    residual_visibilities = prepared["residual_visibilities"]
    sector_images = list(prepared["sector_images"])
    if image_cube_result is None:
        if sector["make_image_cube"]:
            raise ValueError("image-cube finalization requires image cube outputs")
        image_cube_result = make_image_sector_cubes(sector, pipeline_working_dir)
    image_cubes = list(image_cube_result["image_cubes"])
    image_cube_beams = list(image_cube_result["image_cube_beams"])
    image_cube_frequencies = list(image_cube_result["image_cube_frequencies"])
    skymodel_nonpb = prepared["skymodel_nonpb"]
    skymodel_pb = prepared["skymodel_pb"]
    filtered_true_sky = filtered["filtered_true_sky"]
    filtered_apparent_sky = filtered["filtered_apparent_sky"]
    flat_noise_rms = filtered["flat_noise_rms"]
    true_sky_rms = filtered["true_sky_rms"]
    source_catalog = filtered["source_catalog"]
    source_filtering_mask = filtered["source_filtering_mask"]
    diagnostics = diagnostics_result["diagnostics"]
    offsets = diagnostics_result["offsets"]
    diagnostic_plots = list(diagnostics_result["diagnostic_plots"])

    skymodel_image = make_filtered_model_image(
        sector,
        filtered_apparent_sky,
        pb_image,
    )

    if "I" in str(sector["pol"]).upper():
        sector_images.append(
            make_astrometry_corrected_image_record(pb_image, region_record, offsets)
        )
    if config.publish_postage_stamp_previews:
        publish_fits_postage_stamp_artifacts(
            pb_image,
            source_catalog,
            pipeline_working_dir,
            max_sources=config.postage_stamp_preview_count,
            stamp_size_px=config.postage_stamp_preview_size_px,
            clip_percentile=config.fits_preview_clip_percentile,
        )

    output_sector_images = sector_images
    output_extra_images = extra_images
    if sector["compress_images"]:
        output_sector_images, output_extra_images = compress_image_records(
            image_name,
            sector_images,
            extra_images,
            pipeline_working_dir,
            config,
            shell_operation_cls=shell_operation_cls,
        )

    if sector["normalize_flux_scale"]:
        if normalization_result is None:
            raise ValueError("normalize_flux_scale finalization requires normalization outputs")
        normalization_source_catalog = normalization_result["source_catalog"]
        normalize_h5parm = normalization_result["normalize_h5parm"]
    else:
        normalization_source_catalog = None
        normalize_h5parm = None

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
    if residual_visibilities is not None:
        result["residual_visibilities"] = residual_visibilities
    fits_records = (
        output_sector_images
        + output_extra_images
        + [flat_noise_rms, true_sky_rms, source_catalog]
        + ([source_filtering_mask] if source_filtering_mask is not None else [])
        + ([skymodel_image] if skymodel_image is not None else [])
        + image_cubes
    )
    if config.publish_fits_previews:
        publish_fits_image_artifacts(
            fits_records,
            pipeline_working_dir,
            clip_percentile=config.fits_preview_clip_percentile,
        )
    return result


def run_image_sector(
    sector: ImageSectorPayload,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run one imaging sector sequentially."""
    prepared = prepare_image_sector(
        sector,
        pipeline_working_dir,
        execution_config=execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    filtered = filter_image_sector_skymodel(
        sector,
        prepared,
        pipeline_working_dir,
        execution_config=execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    diagnostics_result = calculate_image_sector_diagnostics(
        sector,
        prepared,
        filtered,
        pipeline_working_dir,
    )
    image_cube_result = make_image_sector_cubes(sector, pipeline_working_dir)
    normalization_result = normalize_image_sector_flux_scale(
        sector,
        prepared,
        image_cube_result,
        execution_config=execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    return finalize_image_sector(
        sector,
        prepared,
        filtered,
        diagnostics_result,
        pipeline_working_dir,
        image_cube_result=image_cube_result,
        normalization_result=normalization_result,
        execution_config=execution_config,
        shell_operation_cls=shell_operation_cls,
    )
