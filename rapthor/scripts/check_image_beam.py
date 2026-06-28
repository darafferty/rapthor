#!/usr/bin/env python3
"""
Script to ensure that valid beam information is present in the image header
"""

from argparse import ArgumentParser, RawTextHelpFormatter

from rapthor.execution.image.beam import ensure_image_beam


def main(fits_image_filename: str, beam_size_arcsec: float) -> None:
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
    ensure_image_beam(fits_image_filename, beam_size_arcsec)


if __name__ == "__main__":
    descriptiontext = "Ensure that valid beam information is present in the image header.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("fits_image_filename", type=str, help="Filename of input FITS image")
    parser.add_argument(
        "beam_size_arcsec",
        type=float,
        help="Beam size in arcsec to use for missing beam values",
    )
    args = parser.parse_args()
    main(args.fits_image_filename, args.beam_size_arcsec)
