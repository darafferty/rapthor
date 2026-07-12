"""Image-sector output filename patterns, discovery, and product helpers."""

import multiprocessing
import os
from pathlib import Path
from typing import Mapping, Optional

from rapthor.execution.config import ExecutionConfig
from rapthor.execution.image.commands import (
    build_compress_sector_images_command,
    build_filter_skymodel_command,
    build_make_catalog_from_image_cube_command,
)
from rapthor.execution.image.cubes import make_image_cube
from rapthor.execution.image.flux_normalization import normalize_flux_scale
from rapthor.execution.image.payloads import ImageCubeSpecPayload, ImageSectorPayload
from rapthor.execution.image.restoration import restore_skymodel
from rapthor.execution.image.skymodel_filter import filter_image_skymodel
from rapthor.execution.outputs import (
    compressed_file_record,
    file_records_for_patterns,
    file_records_for_required_patterns,
    require_file,
)
from rapthor.execution.shell import run_external_command
from rapthor.lib.records import file_record


def mfs_non_pb_image_patterns(image_name: str, pipeline_working_dir: str) -> list[str]:
    """Return the possible WSClean MFS non-primary-beam image patterns."""
    return [
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-image.fits"),
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-I-image.fits"),
    ]


def mfs_pb_image_patterns(image_name: str, pipeline_working_dir: str) -> list[str]:
    """Return the possible WSClean MFS primary-beam image patterns."""
    return [
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-image-pb.fits"),
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-I-image-pb.fits"),
    ]


def mfs_extra_image_patterns(
    image_name: str,
    pipeline_working_dir: str,
    *,
    compressed: bool = False,
) -> list[str]:
    """Return glob patterns for optional supplementary WSClean MFS images."""
    suffix = ".fz" if compressed else ""
    return [
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-[QUV]-image.fits{suffix}"),
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-[QUV]-image-pb.fits{suffix}"),
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-*residual.fits{suffix}"),
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-*model-pb.fits{suffix}"),
        os.path.join(pipeline_working_dir, f"{image_name}-MFS-*dirty.fits{suffix}"),
    ]


def channel_image_patterns(image_name: str, stokes: str, pipeline_working_dir: str) -> list[str]:
    """Return channel image patterns for one Stokes image cube."""
    if stokes == "I":
        return [
            os.path.join(pipeline_working_dir, f"{image_name}-0???-image-pb.fits"),
            os.path.join(pipeline_working_dir, f"{image_name}-0???-I-image-pb.fits"),
        ]
    return [os.path.join(pipeline_working_dir, f"{image_name}-0???-{stokes}-image-pb.fits")]


def compress_image_records(
    image_name: str,
    sector_images: list[dict],
    extra_images: list[dict],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    shell_operation_cls=None,
) -> tuple[list[dict], list[dict]]:
    """Compress sector image records and return records for the compressed files."""
    image_paths = [record["path"] for record in sector_images + extra_images]
    sparse_model_images = [path for path in image_paths if _is_sparse_model_image(path)]
    regular_images = [path for path in image_paths if path not in sparse_model_images]
    _compress_images(
        regular_images,
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )
    _compress_images(
        sparse_model_images,
        pipeline_working_dir,
        execution_config,
        lossless=True,
        shell_operation_cls=shell_operation_cls,
    )
    descriptions = [
        "Compressed WSClean non-PB image",
        "Compressed WSClean PB image",
        "Compressed WSClean astrometry-corrected PB image",
    ]
    compressed_sector_images = [
        compressed_file_record(
            record,
            descriptions[index] if index < len(descriptions) else "Compressed WSClean sector image",
        )
        for index, record in enumerate(sector_images)
    ]
    compressed_extra_images = file_records_for_patterns(
        mfs_extra_image_patterns(image_name, pipeline_working_dir, compressed=True)
    )
    return compressed_sector_images, compressed_extra_images


def make_image_cube_records(
    image_name: str,
    image_cube_specs: list[ImageCubeSpecPayload],
    pipeline_working_dir: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build requested image cubes and return image, beam, and frequency records."""
    image_cubes = []
    image_cube_beams = []
    image_cube_frequencies = []
    for spec in image_cube_specs:
        stokes = str(spec["pol"])
        image_cube_filename = str(spec["filename"])
        channel_images = file_records_for_required_patterns(
            channel_image_patterns(image_name, stokes, pipeline_working_dir),
            f"WSClean Stokes-{stokes} channel images",
        )
        image_cube_path = os.path.join(pipeline_working_dir, image_cube_filename)
        make_image_cube([record["path"] for record in channel_images], image_cube_path)
        image_cubes.append(require_file(image_cube_path, f"Stokes-{stokes} image cube"))
        image_cube_beams.append(
            require_file(f"{image_cube_path}_beams.txt", f"Stokes-{stokes} image cube beams")
        )
        image_cube_frequencies.append(
            require_file(
                f"{image_cube_path}_frequencies.txt",
                f"Stokes-{stokes} image cube frequencies",
            )
        )
    return image_cubes, image_cube_beams, image_cube_frequencies


def make_image_cube_catalog_record(
    image_cube: dict,
    image_cube_beams: dict,
    image_cube_frequencies: dict,
    sector: ImageSectorPayload,
    config: ExecutionConfig,
    shell_operation_cls=None,
) -> dict:
    """Build the source catalog used for flux-scale normalization."""
    run_external_command(
        build_make_catalog_from_image_cube_command(
            image_cube["path"],
            image_cube_beams["path"],
            image_cube_frequencies["path"],
            str(sector["output_source_catalog_path"]),
            float(sector["threshisl"]),
            float(sector["threshpix"]),
            int(sector["max_threads"]),
        ),
        str(Path(image_cube["path"]).parent),
        config,
        name="make_catalog_from_image_cube",
        shell_operation_cls=shell_operation_cls,
    )
    return require_file(str(sector["output_source_catalog_path"]), "Normalization source catalog")


def make_normalization_record(
    source_catalog: dict,
    concat_record: dict,
    sector: ImageSectorPayload,
) -> dict:
    """Build the flux-scale normalization h5parm from a source catalog."""
    normalize_flux_scale(
        source_catalog["path"],
        concat_record["path"],
        str(sector["output_normalize_h5parm_path"]),
        reference_skymodels=sector["normalization_skymodels"],
        reference_skymodels_frequencies=sector["normalization_reference_frequencies"],
    )
    return require_file(
        str(sector["output_normalize_h5parm_path"]), "Flux-scale normalization h5parm"
    )


def source_list_records(
    sector: ImageSectorPayload,
    image_name: str,
    pipeline_working_dir: str,
) -> tuple[Optional[dict], Optional[dict]]:
    """Return WSClean source-list records when source-list saving is enabled."""
    if not sector["save_source_list"]:
        return None, None
    skymodel_nonpb = require_file(
        os.path.join(pipeline_working_dir, f"{image_name}-sources.txt"),
        "WSClean apparent-sky source list",
    )
    skymodel_pb = require_file(
        os.path.join(pipeline_working_dir, f"{image_name}-sources-pb.txt"),
        "WSClean true-sky source list",
    )
    return skymodel_nonpb, skymodel_pb


def filter_skymodel_products(
    sector: ImageSectorPayload,
    image_name: str,
    nonpb_image: Mapping[str, str],
    pb_image: Mapping[str, str],
    skymodel_nonpb: Optional[Mapping[str, str]],
    skymodel_pb: Optional[Mapping[str, str]],
    pipeline_working_dir: str,
    execution_config: Optional[ExecutionConfig] = None,
    shell_operation_cls=None,
) -> tuple[dict, dict, dict, dict, dict, dict, Optional[dict]]:
    """Filter skymodel products and return image diagnostics inputs."""
    true_sky_skymodel = skymodel_pb["path"] if sector["save_source_list"] else "none"
    apparent_sky_skymodel = skymodel_nonpb["path"] if sector["save_source_list"] else "none"
    output_root = os.path.join(pipeline_working_dir, image_name)
    beam_ms = list(sector["obs_original_paths"])
    bright_true_sky_skymodel = sector.get("bright_skymodel_pb")
    threshisl = float(sector["threshisl"])
    threshpix = float(sector["threshpix"])
    filter_by_mask = bool(sector["filter_by_mask"])
    source_finder = str(sector["source_finder"])
    ncores = int(sector.get("filter_skymodel_ncores", sector["max_threads"]))

    if _filter_skymodel_needs_subprocess(source_finder, ncores):
        config = execution_config or ExecutionConfig(task_runner="sync")
        run_external_command(
            build_filter_skymodel_command(
                nonpb_image["path"],
                pb_image["path"],
                true_sky_skymodel,
                apparent_sky_skymodel,
                output_root,
                str(sector["vertices_file"]),
                beam_ms,
                threshisl,
                threshpix,
                filter_by_mask,
                source_finder,
                ncores,
                bright_true_sky_skymodel=bright_true_sky_skymodel,
            ),
            pipeline_working_dir,
            config,
            name="filter_skymodel",
            shell_operation_cls=shell_operation_cls,
        )
    else:
        filter_image_skymodel(
            nonpb_image["path"],
            pb_image["path"],
            true_sky_skymodel,
            apparent_sky_skymodel,
            output_root,
            str(sector["vertices_file"]),
            beam_ms,
            bright_true_sky_skymodel=bright_true_sky_skymodel,
            threshisl=threshisl,
            threshpix=threshpix,
            filter_by_mask=filter_by_mask,
            source_finder=source_finder,
            ncores=ncores,
        )

    filtered_true_sky = require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.true_sky.txt"),
        "Filtered true-sky skymodel",
    )
    filtered_apparent_sky = require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.apparent_sky.txt"),
        "Filtered apparent-sky skymodel",
    )
    diagnostics = require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.image_diagnostics.json"),
        "Image diagnostics",
    )
    flat_noise_rms = require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.flat_noise_rms.fits"),
        "Flat-noise RMS image",
    )
    true_sky_rms = require_file(
        os.path.join(pipeline_working_dir, f"{image_name}.true_sky_rms.fits"),
        "True-sky RMS image",
    )
    source_catalog = require_file(
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
    return (
        filtered_true_sky,
        filtered_apparent_sky,
        diagnostics,
        flat_noise_rms,
        true_sky_rms,
        source_catalog,
        source_filtering_mask,
    )


def make_filtered_model_image(
    sector: ImageSectorPayload,
    filtered_apparent_sky: Mapping[str, str],
    pb_image: Mapping[str, str],
) -> Optional[dict]:
    """Build a FITS model image from the filtered apparent skymodel when requested."""
    if not sector["save_filtered_model_image"]:
        return None
    restore_skymodel(
        Path(filtered_apparent_sky["path"]),
        Path(pb_image["path"]),
        Path(str(sector["filtered_model_image_path"])),
    )
    return require_file(str(sector["filtered_model_image_path"]), "Filtered skymodel image")


def _is_sparse_model_image(path: str) -> bool:
    """Return whether ``path`` is a sparse WSClean model image."""
    return os.path.basename(path).endswith("model-pb.fits")


def _compress_images(
    images: list[str],
    pipeline_working_dir: str,
    execution_config: ExecutionConfig,
    *,
    lossless: bool = False,
    shell_operation_cls=None,
) -> None:
    if not images:
        return
    run_external_command(
        build_compress_sector_images_command(images, lossless=lossless),
        pipeline_working_dir,
        execution_config,
        shell_operation_cls=shell_operation_cls,
    )


def _current_process_is_daemon() -> bool:
    """Return whether the current process is a daemonic worker."""
    return bool(multiprocessing.current_process().daemon)


def _filter_skymodel_needs_subprocess(source_finder: str, ncores: int) -> bool:
    """Return whether source filtering should be isolated from the task process."""
    if _current_process_is_daemon():
        return True
    return source_finder.lower() == "bdsf" and ncores != 1
