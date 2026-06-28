"""Image-sector output filename pattern helpers."""

import os


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
