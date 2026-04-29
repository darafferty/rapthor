#!/usr/bin/env python3
"""
Script to apply astrometry corrections to an image made using faceting
"""

import json
import logging
import shutil
import subprocess
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

import numpy as np
import scipy.ndimage as nd
from astropy.io.fits import writeto as fits_write
from lsmtool.utils import rasterize

from rapthor.lib.facet import read_ds9_region_file
from rapthor.lib.fitsimage import FITSImage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_corrections(corrections: dict) -> dict:
    """
    Validates that an astrometry corrections dict has the expected structure

    Parameters
    ----------
    corrections : dict
        Astrometry corrections dict to be validated

    Returns
    -------
    validated_corrections : dict
        Validated astrometry corrections dict
    """
    required_keys = [
        "facet_name",
        "meanRAOffsetDeg",
        "meanDecOffsetDeg",
        "stdRAOffsetDeg",
        "stdDecOffsetDeg",
    ]

    for key in required_keys:
        if key not in corrections:
            raise ValueError(f"Missing key in corrections dict: {key}")

    if not len({len(item) for item in corrections.values()}) == 1:
        raise ValueError("Corrections should have equal length")

    return corrections


def main(
    input_image: Path,
    region_file: Path,
    corrections_file: Path,
    output_image: Path,
    overwrite: bool,
) -> int:
    """
    Applies astrometry corrections to a FITS image

    Corrections are done by cutting out facet images and shifting them by the given offsets in RA
    and Dec. The facet cutouts are then added back together to produce the full, corrected image.
    The facet image cutouts are padded before shifting to avoid gaps between the facets, averaging
    the values in regions with contributions from more than one facet

    Parameters
    ----------
    input_image : Path
        Filename of uncorrected input FITS image
    region_file : Path
        Filename of input ds9 region file that defines the facets
    corrections_file : Path
        Filename of input JSON file that contains the corrections to apply. If None, no
        corrections are applied, and the input image is just copied to the output image
    output_image : Path or None
        Filename of corrected output FITS image. If None, the filename is constructed from
        input_image by adding the infix "-ast" before the extensions ".fits" or ".fits.fz" (if
        present). If these extensions are not present, the filename is constructed by adding
        "-ast.fits" to the end of input_image (or "-ast.fits.fz" if input_image ends with ".fz",
        indicating compression with fpack)
    overwrite : bool
        If True, overwrite existing output file
    """
    # Check inputs
    if output_image is None:
        # Construct the output image filename by adding the infix "-ast" before ".fits" (if
        # present). Note that if the input image is compressed with fpack (as indicated by the
        # extension ".fz"), the output filename does not include this extension, as it must be
        # written without compression. In this case, compression is done with a call to fpack
        root = input_image.name
        if root.endswith(".fz"):
            root = root[:-3]
        if root.endswith(".fits"):
            root = root[:-5]
        output_image = Path(f"{root}-ast.fits")
    if corrections_file is None:
        # No correction possible: copy input file to output and return
        if input_image.suffix == ".fz" and output_image.suffix != ".fz":
            output_image = output_image.with_suffix(output_image.suffix + ".fz")
        if not output_image.exists() or overwrite:
            shutil.copy(input_image, output_image)
        return

    # Read in corrections and check they have the expected structure
    with open(corrections_file, "r") as fp:
        corrections = validate_corrections(json.load(fp))

    # Read in the input image and construct the output arrays
    uncorrected_image = FITSImage(input_image)
    wcs = uncorrected_image.get_wcs()
    ra_scale = wcs.wcs.cdelt[0]  # degrees / pix
    dec_scale = wcs.wcs.cdelt[1]  # degrees / pix
    corrected_data = np.zeros_like(uncorrected_image.img_data)  # flattened, 2-D image
    sum_map = np.zeros_like(uncorrected_image.img_data)

    # Read in the facets, using the WCS from the input image to keep the coordinate transformations
    # the same
    facets = read_ds9_region_file(region_file, wcs=wcs)

    # Loop over the facets, applying the corrections
    facet_map = {name: i for i, name in enumerate(corrections["facet_name"])}
    for idx, facet in enumerate(facets):
        logger.info("Processing facet %d/%d", idx, len(facets))

        # Pad the facet polygon by 2 pixels
        poly_padded = facet.polygon.buffer(2)
        vertices = list(
            zip(
                poly_padded.exterior.coords.xy[0].tolist(),
                poly_padded.exterior.coords.xy[1].tolist(),
            )
        )

        # Blank pixels outside of the polygon with zeros
        facet_data = rasterize(vertices, uncorrected_image.img_data.copy())

        # Apply corrections (all values in pixels)
        #
        # Note: The offsets are defined as (LOFAR model value) - (comparison model value); e.g., a
        # positive Dec offset indicates that the LOFAR sources are on average North of the
        # comparison source positions. So we negate the offsets during correction
        if facet.name not in corrections["facet_name"]:
            logger.warning(
                f"Astrometry offsets for Facet {facet.name} not found. No corrections will "
                "be done for this facet"
            )
        else:
            facet_index = facet_map[facet.name]
            ra_correction = -corrections["meanRAOffsetDeg"][facet_index] / ra_scale
            dec_correction = -corrections["meanDecOffsetDeg"][facet_index] / dec_scale
            ra_correction_std = corrections["stdRAOffsetDeg"][facet_index] / ra_scale
            dec_correction_std = corrections["stdDecOffsetDeg"][facet_index] / dec_scale
            total_correction = np.hypot(ra_correction, dec_correction)
            total_error = np.hypot(ra_correction_std, dec_correction_std)
            if total_correction > total_error:
                facet_data = nd.shift(facet_data, [dec_correction, ra_correction], order=3)
                logger.info(
                    "Corrected facet %s by %.3f arcsec in RA and %.3f arcsec in Dec",
                    facet.name,
                    ra_correction * ra_scale * 3600,
                    dec_correction * dec_scale * 3600,
                )
            else:
                logger.info(
                    "Skipping correction for facet %s since total shift (%.3f +/- %.3f arcsec) "
                    "is not significant",
                    facet.name,
                    total_correction * ra_scale * 3600,
                    total_error * dec_scale * 3600,
                )

        # Reduce padding to avoid any edge effects
        poly_padded = facet.polygon.buffer(1)
        vertices = list(
            zip(
                poly_padded.exterior.coords.xy[0].tolist(),
                poly_padded.exterior.coords.xy[1].tolist(),
            )
        )

        # Add facet data to corrected image
        facet_mask = rasterize(vertices, np.ones_like(facet_data))
        corrected_data += facet_mask * facet_data
        sum_map += facet_mask

    # Divide the sum map into the corrected image using a mask to exclude any blank pixels
    mask = sum_map > 0
    corrected_data[mask] /= sum_map[mask]

    # Adjust the shape of the corrected data to match the number of axes of the original image
    output_shape = [1 for axis in range(uncorrected_image.header["NAXIS"] - 2)] + list(
        corrected_data.shape
    )

    # Write out the corrected image (always uncompressed, as astropy.io.fits.writeto does not
    # support fpack compression)
    fits_write(
        output_image,
        data=corrected_data.reshape(output_shape),
        header=uncorrected_image.header,
        overwrite=overwrite,
    )

    # Compress with fpack if input image was compressed. fpack adds the extension ".fz" to the file
    if input_image.suffix == ".fz":
        cmd = ["fpack", output_image]
        try:
            result = subprocess.run(cmd, check=True)
            output_image.unlink()  # remove uncompressed version
            return result.returncode
        except subprocess.CalledProcessError as err:
            print(err, file=sys.stderr)
            return err.returncode


if __name__ == "__main__":
    descriptiontext = "Apply astrometry corrections to an image.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("input_image", help="Filename of uncorrected input FITS image", type=Path)
    parser.add_argument("region_file", help="Filename of input ds9 region file", type=Path)
    parser.add_argument(
        "--corrections_file",
        default=None,
        help="Filename of input json file with astrometry corrections",
        type=Path,
    )
    parser.add_argument(
        "--output_image", default=None, help="Filename of corrected output FITS image", type=Path
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite an existing output image file",
    )
    args = parser.parse_args()

    try:
        main(
            args.input_image,
            args.region_file,
            args.corrections_file,
            args.output_image,
            args.overwrite,
        )
    except ValueError as e:
        log = logging.getLogger("rapthor:correct_astrometry")
        log.critical(e)
