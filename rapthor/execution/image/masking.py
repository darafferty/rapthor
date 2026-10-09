"""Image mask creation helpers."""

import logging
from typing import Optional, Sequence, Union

from lsmtool.utils import rasterize_polygon_mask_exterior

from rapthor.lib import miscellaneous as misc

log = logging.getLogger("rapthor:image:masking")


def blank_image(
    output_image: str,
    input_image: Optional[str] = None,
    vertices_file: Optional[str] = None,
    reference_ra_deg: Optional[float] = None,
    reference_dec_deg: Optional[float] = None,
    cellsize_deg: Optional[float] = None,
    imsize: Optional[Union[str, Sequence[int]]] = None,
) -> None:
    """
    Create a mask image and optionally blank everything outside a polygon.

    When ``input_image`` is omitted, a new FITS template is created with value
    one everywhere. When ``vertices_file`` is supplied, the output is then
    masked using the polygon boundary in that vertices file. When
    ``input_image`` is supplied, the polygon mask is applied to that image and
    written to ``output_image``.
    """
    make_template = input_image is None
    if make_template:
        log.info("Input image not given. Making empty image...")
        ximsize, yimsize = _parse_image_size(imsize)
        if reference_ra_deg is None or reference_dec_deg is None or cellsize_deg is None:
            raise ValueError(
                "reference_ra_deg, reference_dec_deg, and cellsize_deg are required "
                "when input_image is not given"
            )
        misc.make_template_image(
            output_image,
            float(reference_ra_deg),
            float(reference_dec_deg),
            ximsize=ximsize,
            yimsize=yimsize,
            cellsize_deg=float(cellsize_deg),
            fill_val=1,
        )

    if vertices_file is not None:
        source_image = output_image if make_template else input_image
        rasterize_polygon_mask_exterior(source_image, vertices_file, output_image)


def _parse_image_size(imsize: Optional[Union[str, Sequence[int]]]) -> tuple[int, int]:
    """Return ``(x_size, y_size)`` from a comma-separated string or two-item sequence."""
    if imsize is None:
        raise ValueError("imsize is required when input_image is not given")

    if isinstance(imsize, str):
        parts = [part.strip() for part in imsize.split(",")]
    else:
        parts = list(imsize)

    if len(parts) != 2:
        raise ValueError("imsize must contain exactly two values: x_size,y_size")
    return int(parts[0]), int(parts[1])
