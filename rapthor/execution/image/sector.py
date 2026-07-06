"""Execution helpers for one image sector."""

from typing import Optional

from rapthor.execution.artifacts import publish_fits_image_artifacts
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


def run_image_sector(
    sector: ImageSectorPayload,
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run one imaging sector."""
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

    image_cubes = []
    image_cube_beams = []
    image_cube_frequencies = []
    if sector["make_image_cube"]:
        image_cubes, image_cube_beams, image_cube_frequencies = make_image_cube_records(
            image_name,
            list(sector["image_cube_specs"]),
            pipeline_working_dir,
        )

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
        nonpb_image,
        pb_image,
        skymodel_nonpb,
        skymodel_pb,
        pipeline_working_dir,
        execution_config=config,
        shell_operation_cls=shell_operation_cls,
    )

    skymodel_image = make_filtered_model_image(
        sector,
        filtered_apparent_sky,
        pb_image,
    )

    diagnostics, offsets, diagnostic_plots = run_image_diagnostics(
        sector,
        image_name,
        nonpb_image,
        pb_image,
        flat_noise_rms,
        true_sky_rms,
        source_catalog,
        diagnostics,
        region_record,
        pipeline_working_dir,
    )
    if "I" in str(sector["pol"]).upper():
        sector_images.append(
            make_astrometry_corrected_image_record(pb_image, region_record, offsets)
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

    normalization_source_catalog = None
    normalize_h5parm = None
    if sector["normalize_flux_scale"]:
        normalization_source_catalog, normalize_h5parm = make_normalization_records(
            image_cubes[0],
            image_cube_beams[0],
            image_cube_frequencies[0],
            concat_record,
            sector,
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
        publish_fits_image_artifacts(fits_records, pipeline_working_dir)
    return result
