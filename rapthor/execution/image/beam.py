"""FITS beam-header helpers for image execution."""

from astropy.io import fits


def ensure_image_beam(fits_image_filename: str, beam_size_arcsec: float) -> None:
    """
    Ensure that valid beam information is present in a FITS image header.

    If no valid beam information is present, missing or non-positive beam axes
    are replaced with a circular beam using ``beam_size_arcsec``. Non-positive
    fallback beam sizes are replaced with a conservative 10 arcsec default.
    """
    if beam_size_arcsec <= 0.0:
        beam_size_arcsec = 10.0

    with fits.open(fits_image_filename, mode="update") as hdu:
        header = hdu[0].header
        if "BMAJ" not in header or header["BMAJ"] <= 0.0:
            header["BMAJ"] = beam_size_arcsec / 3600
        if "BMIN" not in header or header["BMIN"] <= 0.0:
            header["BMIN"] = beam_size_arcsec / 3600
        if "BPA" not in header:
            header["BPA"] = 0.0
