"""Execution helpers for one image sector."""

from typing import Mapping, Optional

from rapthor.execution.artifacts import (
    publish_fits_image_artifacts,
    publish_fits_postage_stamp_artifacts,
)
from rapthor.execution.config import ExecutionConfig
from rapthor.execution.image.astrometry import make_astrometry_corrected_image_record
from rapthor.execution.image.diagnostics import run_image_diagnostics
from rapthor.execution.image.outputs import (
    compress_image_records,
    filter_skymodel_products,
    make_filtered_model_image,
    make_image_cube_catalog_record,
    make_image_cube_records,
    make_normalization_record,
    mfs_extra_image_patterns,
    source_list_records,
)
from rapthor.execution.image.payloads import ImageSectorPayload
from rapthor.execution.image.preparation import (
    concatenate_prepared_visibilities,
    ensure_facet_region,
    ensure_imaging_mask,
    prepare_and_concatenate_visibilities,
    prepare_visibility_ms,
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

    prepared_records, concat_record = prepare_and_concatenate_visibilities(
        sector,
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )
    image_products = make_image_sector_wsclean_products(
        sector,
        concat_record,
        pipeline_working_dir,
        execution_config=config,
        shell_operation_cls=shell_operation_cls,
    )
    image_products = finish_image_sector_wsclean_products(
        sector,
        image_products,
        pipeline_working_dir,
        execution_config=config,
        shell_operation_cls=shell_operation_cls,
    )
    residual_result = make_image_sector_residual_visibility_product(
        sector,
        concat_record,
        image_products,
        pipeline_working_dir,
        execution_config=config,
        shell_operation_cls=shell_operation_cls,
    )
    return assemble_image_sector_preparation(
        prepared_records,
        concat_record,
        image_products,
        residual_result,
    )


def prepare_image_sector_visibility(
    sector: ImageSectorPayload,
    prepare_task: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Prepare one observation Measurement Set for an image sector."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    return prepare_visibility_ms(
        sector,
        prepare_task,
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )


def concatenate_image_sector_visibilities(
    sector: ImageSectorPayload,
    prepared_records: list[dict],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Concatenate prepared observation Measurement Sets for an image sector."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    return concatenate_prepared_visibilities(
        sector,
        prepared_records,
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )


def make_image_sector_wsclean_products(
    sector: ImageSectorPayload,
    concat_record: Mapping[str, str],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Run or reuse WSClean products for one image sector."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    image_name = str(sector["image_name"])
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
    skymodel_nonpb, skymodel_pb = source_list_records(
        sector,
        image_name,
        pipeline_working_dir,
    )
    return {
        "mask_record": mask_record,
        "region_record": region_record,
        "nonpb_image": nonpb_image,
        "pb_image": pb_image,
        "wsclean_ran": wsclean_ran,
        "extra_images": extra_images,
        "sector_images": [nonpb_image, pb_image],
        "skymodel_nonpb": skymodel_nonpb,
        "skymodel_pb": skymodel_pb,
    }


def finish_image_sector_wsclean_products(
    sector: ImageSectorPayload,
    image_products: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Restore bright sources when requested and validate new WSClean beams."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    nonpb_image, pb_image = restore_bright_source_images(
        sector,
        image_products["nonpb_image"],
        image_products["pb_image"],
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )
    if image_products["wsclean_ran"]:
        check_wsclean_beams(
            (pb_image, nonpb_image),
            sector,
        )
    return dict(image_products) | {
        "nonpb_image": nonpb_image,
        "pb_image": pb_image,
        "sector_images": [nonpb_image, pb_image],
    }


def make_image_sector_residual_visibility_product(
    sector: ImageSectorPayload,
    concat_record: Mapping[str, str],
    image_products: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Create residual visibilities when the sector requests them."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    residual_visibilities = make_residual_visibility_record(
        sector,
        concat_record,
        pipeline_working_dir,
        config,
        force=bool(image_products["wsclean_ran"]),
        shell_operation_cls=shell_operation_cls,
    )
    return {"residual_visibilities": residual_visibilities}


def assemble_image_sector_preparation(
    prepared_records: list[dict],
    concat_record: Mapping[str, str],
    image_products: Mapping[str, object],
    residual_result: Optional[Mapping[str, object]] = None,
) -> dict:
    """Assemble the prepared sector payload consumed by downstream image tasks."""
    residual_visibilities = None
    if residual_result is not None:
        residual_visibilities = residual_result["residual_visibilities"]
    return {
        "prepared_records": prepared_records,
        "concat_record": concat_record,
        "mask_record": image_products["mask_record"],
        "region_record": image_products["region_record"],
        "nonpb_image": image_products["nonpb_image"],
        "pb_image": image_products["pb_image"],
        "wsclean_ran": image_products["wsclean_ran"],
        "extra_images": image_products["extra_images"],
        "residual_visibilities": residual_visibilities,
        "sector_images": image_products["sector_images"],
        "skymodel_nonpb": image_products["skymodel_nonpb"],
        "skymodel_pb": image_products["skymodel_pb"],
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


def make_image_sector_cube_catalog(
    sector: ImageSectorPayload,
    image_cube_result: Mapping[str, object],
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Build the source catalog used for one sector flux-scale normalization."""
    if not sector["normalize_flux_scale"]:
        return {"source_catalog": None}

    image_cubes = list(image_cube_result["image_cubes"])
    image_cube_beams = list(image_cube_result["image_cube_beams"])
    image_cube_frequencies = list(image_cube_result["image_cube_frequencies"])
    if not image_cubes:
        raise ValueError("make_catalog_from_image_cube requires image cube outputs")

    config = execution_config or ExecutionConfig(task_runner="sync")
    return {
        "source_catalog": make_image_cube_catalog_record(
            image_cubes[0],
            image_cube_beams[0],
            image_cube_frequencies[0],
            sector,
            config,
            shell_operation_cls=shell_operation_cls,
        )
    }


def normalize_image_sector_flux_scale(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    catalog_result: Mapping[str, object],
) -> dict:
    """Build a flux-scale normalization h5parm for one sector when requested."""
    if not sector["normalize_flux_scale"]:
        return {"normalize_h5parm": None}

    source_catalog = catalog_result["source_catalog"]
    if source_catalog is None:
        raise ValueError("normalize_flux_scale requires a source catalog")

    return {
        "normalize_h5parm": make_normalization_record(
            source_catalog,
            prepared["concat_record"],
            sector,
        )
    }


def restore_image_sector_skymodel(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    filtered: Mapping[str, object],
) -> dict:
    """Build the filtered skymodel image for one sector when requested."""
    return {
        "skymodel_image": make_filtered_model_image(
            sector,
            filtered["filtered_apparent_sky"],
            prepared["pb_image"],
        )
    }


def _sector_image_records_with_astrometry(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    diagnostics_result: Mapping[str, object],
) -> tuple[list[dict], list[dict]]:
    """Return sector image records after adding astrometry-corrected images."""
    sector_images = list(prepared["sector_images"])
    if "I" in str(sector["pol"]).upper():
        sector_images.append(
            make_astrometry_corrected_image_record(
                prepared["pb_image"],
                prepared["region_record"],
                diagnostics_result["offsets"],
            )
        )
    return sector_images, list(prepared["extra_images"])


def compress_image_sector_products(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    diagnostics_result: Mapping[str, object],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Compress one sector's image products when requested."""
    if not sector["compress_images"]:
        return {"sector_images": None, "extra_images": None}

    config = execution_config or ExecutionConfig(task_runner="sync")
    sector_images, extra_images = _sector_image_records_with_astrometry(
        sector,
        prepared,
        diagnostics_result,
    )
    output_sector_images, output_extra_images = compress_image_records(
        str(sector["image_name"]),
        sector_images,
        extra_images,
        pipeline_working_dir,
        config,
        shell_operation_cls=shell_operation_cls,
    )
    return {
        "sector_images": output_sector_images,
        "extra_images": output_extra_images,
    }


def finalize_image_sector(
    sector: ImageSectorPayload,
    prepared: Mapping[str, object],
    filtered: Mapping[str, object],
    diagnostics_result: Mapping[str, object],
    pipeline_working_dir: str,
    image_cube_result: Optional[Mapping[str, object]] = None,
    catalog_result: Optional[Mapping[str, object]] = None,
    normalization_result: Optional[Mapping[str, object]] = None,
    restored_model_result: Optional[Mapping[str, object]] = None,
    compression_result: Optional[Mapping[str, object]] = None,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> dict:
    """Build post-WSClean sector products and assemble the output record."""
    config = execution_config or ExecutionConfig(task_runner="sync")
    prepared_records = list(prepared["prepared_records"])
    pb_image = prepared["pb_image"]
    region_record = prepared["region_record"]
    residual_visibilities = prepared["residual_visibilities"]
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

    if config.publish_postage_stamp_previews:
        publish_fits_postage_stamp_artifacts(
            pb_image,
            source_catalog,
            pipeline_working_dir,
            max_sources=config.postage_stamp_preview_count,
            stamp_size_px=config.postage_stamp_preview_size_px,
            clip_percentile=config.fits_preview_clip_percentile,
        )

    if restored_model_result is None:
        if sector["save_filtered_model_image"]:
            raise ValueError("filtered-model finalization requires restored skymodel output")
        restored_model_result = restore_image_sector_skymodel(sector, prepared, filtered)
    skymodel_image = restored_model_result["skymodel_image"]

    if sector["compress_images"]:
        if compression_result is None:
            raise ValueError("compression finalization requires compressed image outputs")
        output_sector_images = list(compression_result["sector_images"])
        output_extra_images = list(compression_result["extra_images"])
    else:
        output_sector_images, output_extra_images = _sector_image_records_with_astrometry(
            sector,
            prepared,
            diagnostics_result,
        )

    if sector["normalize_flux_scale"]:
        if catalog_result is None:
            raise ValueError("normalize_flux_scale finalization requires a source catalog")
        if normalization_result is None:
            raise ValueError("normalize_flux_scale finalization requires normalization outputs")
        normalization_source_catalog = catalog_result["source_catalog"]
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
    catalog_result = make_image_sector_cube_catalog(
        sector,
        image_cube_result,
        execution_config=execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    normalization_result = normalize_image_sector_flux_scale(
        sector,
        prepared,
        catalog_result,
    )
    restored_model_result = restore_image_sector_skymodel(sector, prepared, filtered)
    compression_result = compress_image_sector_products(
        sector,
        prepared,
        diagnostics_result,
        pipeline_working_dir,
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
        catalog_result=catalog_result,
        normalization_result=normalization_result,
        restored_model_result=restored_model_result,
        compression_result=compression_result,
        execution_config=execution_config,
        shell_operation_cls=shell_operation_cls,
    )
