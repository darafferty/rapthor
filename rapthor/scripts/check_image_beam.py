#!/usr/bin/env python3
"""
Script to check that valid beam information is present in the image header
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.io import fits


def main(fits_image_filename, beam_size_arcsec):
    """
    Check that valid beam information is present in the image header

    If no valid beam information is present, the values for a basic circular beam are
    added.

    Parameters
    ----------
    fits_image_filename : str
        Filename of input FITS image
    beam_size_arcsec : float
        The beam size in arcsec to use when no beam information is found in the input
        image. A circular beam is adopted.
    """
    # Check that beam size is valid; if not, set to a typical value
    if beam_size_arcsec <= 0.0:
        beam_size_arcsec = 10.0

    # Open the input HDU (Header Data Unit) and header. Updates to the header will be
    # automatically written to the input file after the exiting the "with" scope
    with fits.open(fits_image_filename, mode="update") as hdu:
        # Replace missing values with ones for a circular beam of size beam_size_arcsec
        # in degrees
        header = hdu[0].header
        if "BMAJ" not in header or header["BMAJ"] <= 0.0:
            header["BMAJ"] = beam_size_arcsec / 3600
        if "BMIN" not in header or header["BMIN"] <= 0.0:
            header["BMIN"] = beam_size_arcsec / 3600
        if 'BPA' not in header:
            header['BPA'] = 0.0


if __name__ == '__main__':
    descriptiontext = (
        "Check that valid beam information is present in the image header.\n"
    )

    parser = ArgumentParser(
        description=descriptiontext, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "fits_image_filename", type=str, help="Filename of input FITS image"
    )
    parser.add_argument(
        "beam_size_arcsec",
        type=float,
        help="Beam size in arcsec to use for missing beam values",
    )
    args = parser.parse_args()
    main(args.fits_image_filename, args.beam_size_arcsec)
