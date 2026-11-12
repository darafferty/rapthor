"""Shared region-file helpers for execution scripts."""

from lsmtool.facet import make_ds9_region_file, read_from_skymodel
from lsmtool.operations_lib import make_wcs
from shapely.geometry import Polygon, box

from rapthor.lib import miscellaneous as misc


def read_facets(skymodel: str, ra_mid: float, dec_mid: float, width_ra: float, width_dec: float):
    """
    Read the facets of a calibration sky model inside a bounding box.

    The bounding box is centred on ``ra_mid`` and ``dec_mid`` in degrees.
    ``width_ra`` is the RA width in degrees corrected to Dec = 0, matching the
    convention expected by ``lsmtool.facet.read_from_skymodel``.
    """
    return read_from_skymodel(
        skymodel,
        ra_mid,
        dec_mid,
        width_ra,
        width_dec,
        wcs=make_wcs(ra_mid, dec_mid, misc.WCS_PIXEL_SCALE),
    )


def make_ds9_region_from_skymodel(
    skymodel: str,
    ra_mid: float,
    dec_mid: float,
    width_ra: float,
    width_dec: float,
    region_file: str,
    *,
    enclose_names: bool = True,
) -> None:
    """
    Make a DS9 facet region file from a calibration sky model.

    The bounding box is that of ``read_facets``.
    """
    facets = read_facets(skymodel, ra_mid, dec_mid, width_ra, width_dec)
    make_ds9_region_file(facets, region_file, enclose_names=enclose_names)


def count_facets_in_image(
    facets,
    *,
    image_ra: float,
    image_dec: float,
    image_width_ra: float,
    image_width_dec: float,
    cellsize_deg: float,
) -> int:
    """
    Count the facets that WSClean images for an image.

    The image is centred on ``image_ra`` and ``image_dec``; its widths and its
    pixel size are in degrees. WSClean leaves out the facets of a region file
    that do not cover a pixel of the image.
    """
    # The image is a rectangle in the projection around its own centre
    wcs = make_wcs(image_ra, image_dec, misc.WCS_PIXEL_SCALE)
    x, y = wcs.wcs_world2pix(image_ra, image_dec, misc.WCS_ORIGIN)
    half_width_x = image_width_ra / misc.WCS_PIXEL_SCALE / 2
    half_width_y = image_width_dec / misc.WCS_PIXEL_SCALE / 2
    image = box(x - half_width_x, y - half_width_y, x + half_width_x, y + half_width_y)
    pixel_area = (cellsize_deg / misc.WCS_PIXEL_SCALE) ** 2
    return sum(
        Polygon(wcs.wcs_world2pix(facet.vertices, misc.WCS_ORIGIN)).intersection(image).area
        > pixel_area
        for facet in facets
    )
