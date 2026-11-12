"""Shared region-file helpers for execution scripts."""

from lsmtool.facet import make_ds9_region_file, read_from_skymodel
from lsmtool.operations_lib import make_wcs
from shapely.geometry import box

from rapthor.lib import miscellaneous as misc


def _read_facets(skymodel: str, ra_mid: float, dec_mid: float, width_ra: float, width_dec: float):
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

    The bounding box is centred on ``ra_mid`` and ``dec_mid`` in degrees.
    ``width_ra`` is the RA width in degrees corrected to Dec = 0, matching the
    convention expected by ``lsmtool.facet.read_from_skymodel``.
    """
    facets = _read_facets(skymodel, ra_mid, dec_mid, width_ra, width_dec)
    make_ds9_region_file(facets, region_file, enclose_names=enclose_names)


def count_facets_in_image(
    skymodel: str,
    ra_mid: float,
    dec_mid: float,
    width_ra: float,
    width_dec: float,
    *,
    image_ra: float,
    image_dec: float,
    image_width_ra: float,
    image_width_dec: float,
) -> int:
    """
    Count the facets of a calibration sky model that overlap an image.

    The first five arguments are those of ``make_ds9_region_from_skymodel``.
    The image is centred on ``image_ra`` and ``image_dec`` and the widths are
    in degrees. WSClean only images the facets of a region file that overlap
    the image, so this is the number of facets that it works on.
    """
    facets = _read_facets(skymodel, ra_mid, dec_mid, width_ra, width_dec)
    x, y = facets[0].wcs.wcs_world2pix(image_ra, image_dec, misc.WCS_ORIGIN)
    half_width_x = image_width_ra / misc.WCS_PIXEL_SCALE / 2
    half_width_y = image_width_dec / misc.WCS_PIXEL_SCALE / 2
    image = box(x - half_width_x, y - half_width_y, x + half_width_x, y + half_width_y)
    return sum(facet.polygon.intersects(image) for facet in facets)
