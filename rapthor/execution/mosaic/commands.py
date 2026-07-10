"""Mosaic command builders."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DrawModelMosaicOptions:
    """WSClean options used to render a model mosaic from sky-model components."""

    skymodel: str
    output_root: str
    ra_dec: list[str]
    frequency_bandwidth: list[object]
    cellsize_deg: object
    imsize: list[int]
    num_threads: int
    num_terms: int = 1


def build_compress_mosaic_command(mosaic_filename: str) -> list[str]:
    """Build the `fpack` command for one mosaic image."""
    return ["fpack", mosaic_filename]


def build_draw_model_mosaic_command(options: DrawModelMosaicOptions) -> list[str]:
    """Build the WSClean command that renders one model mosaic image."""
    return [
        "wsclean",
        "-j",
        str(options.num_threads),
        "-draw-model",
        options.skymodel,
        "-draw-spectral-terms",
        str(options.num_terms),
        "-name",
        options.output_root,
        "-draw-centre",
        *[str(value) for value in options.ra_dec],
        "-draw-frequencies",
        *[str(value) for value in options.frequency_bandwidth],
        "-size",
        *[str(value) for value in options.imsize],
        "-scale",
        str(options.cellsize_deg),
    ]
