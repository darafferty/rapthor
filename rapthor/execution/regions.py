"""Shared region-file helpers for execution scripts."""

from lsmtool.facet import make_ds9_region_file, read_skymodel
from lsmtool.operations_lib import make_wcs

from rapthor.lib import miscellaneous as misc


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
    convention expected by ``lsmtool.facet.read_skymodel``.
    """
    facets = read_skymodel(
        skymodel,
        ra_mid,
        dec_mid,
        width_ra,
        width_dec,
        wcs=make_wcs(ra_mid, dec_mid, misc.WCS_PIXEL_SCALE),
    )
    make_ds9_region_file(facets, region_file, enclose_names=enclose_names)
