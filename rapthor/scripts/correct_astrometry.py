#!/usr/bin/env python3
"""
Script to apply astrometry corrections to an image made using faceting
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.io.fits import writeto as fits_write
import json
import logging
from lsmtool.utils import rasterize
import numpy as np
from pathlib import Path
from rapthor.lib.facet import read_ds9_region_file
from rapthor.lib.fitsimage import FITSImage
import scipy.ndimage as nd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(
    input_image: Path,
    region_file: Path,
    corrections_file: Path,
    output_image: Path,
    overwrite: bool,
):
    """
    Applies astrometry corrections to a FITS image

    Parameters
    ----------
    input_image : str
        Filename of uncorrected input FITS image
    region_file : str
        Filename of input ds9 region file that defines the facets
    corrections_file : str
        Filename of input JSON file that contains the corrections to apply
    output_image : str
        Filename of corrected output FITS image
    overwrite : bool
        If True, overwrite existing output file
    """
    # Read in facets, corrections, and input image
    with open(corrections_file, "r") as fp:
        corrections = json.load(fp)
    uncorrected_image = FITSImage(input_image)
    wcs = uncorrected_image.get_wcs()
    facets = read_ds9_region_file(region_file, wcs=wcs)
    ra_scale = wcs.wcs.cdelt[0]  # degrees / pix
    dec_scale = wcs.wcs.cdelt[1]  # degrees / pix
    corrected_data = np.zeros_like(uncorrected_image.img_data)  # flattened, 2-D image
    sum_map = np.zeros_like(uncorrected_image.img_data)

    # Loop over the facets, applying the corrections
    for facet in facets:
        # Pad the facet polygon by 2 pixels
        poly_padded = facet.polygon.buffer(2)
        vertices = list(
            zip(
                poly_padded.exterior.coords.xy[0].tolist(),
                poly_padded.exterior.coords.xy[1].tolist(),
            )
        )

        # Blank pixels outside of the polygon
        facet_data = rasterize(vertices, uncorrected_image.img_data.copy())

        # Apply corrections (all values in pixels)
        #
        # Note: The offsets are defined as (LOFAR model value) - (comparison model value); e.g., a
        # positive Dec offset indicates that the LOFAR sources are on average North of the
        # comparison source positions. So we negate the offsets during correction
        if facet.name not in corrections["facet_name"]:
            logger.warn(
                f"Astrometry offsets for Facet {facet.name} not found. No corrections will "
                "be done for this facet"
            )
        else:
            facet_index = corrections["facet_name"].index(facet.name)
            ra_correction = -corrections["meanRAOffsetDeg"][facet_index] / ra_scale
            dec_correction = -corrections["meanDecOffsetDeg"][facet_index] / dec_scale
            ra_correction_std = corrections["stdRAOffsetDeg"][facet_index] / ra_scale
            dec_correction_std = corrections["stdDecOffsetDeg"][facet_index] / dec_scale
            total_correction = np.hypot(ra_correction, dec_correction)
            total_error = np.hypot(ra_correction_std, dec_correction_std)
            if total_correction > total_error:
                facet_data = nd.shift(
                    facet_data, [dec_correction, ra_correction], order=3
                )
                logger.info(
                    "Corrected facet %s by %.3f arcsec in RA and %.3f arcsec in Dec",
                    facet.name, ra_correction * ra_scale * 3600, dec_correction * dec_scale * 3600,
                )
            else:
                logger.info(
                    "Skipping correction for facet %s since total shift (%.3f +/- %.3f arcsec) "
                    "is not significant",
                    facet.name,
                    total_correction * 3600,
                    total_error * 3600,
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
        corrected_data += rasterize(vertices, facet_data.copy())
        sum_map += rasterize(vertices, np.ones_like(facet_data))

    # Write out the corrected image
    sum_map[sum_map < 1] = 1
    fits_write(
        output_image,
        data=corrected_data / sum_map,
        header=uncorrected_image.header,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    descriptiontext = "Apply astrometry corrections to an image.\n"

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument('input_image', help='Filename of uncorrected input FITS image', type=Path)
    parser.add_argument('region_file', help='Filename of input ds9 region file', type=Path)
    parser.add_argument('corrections_file', help='Filename of input json file with astrometry corrections', type=Path)
    parser.add_argument('output_image', help='Filename of corrected output FITS image', type=Path)
    parser.add_argument('--overwrite', default=False, action='store_true', help='Overwrite an exising output image file (default=False)')
    args = parser.parse_args()

    try:
        main(args.input_image, args.region_file, args.corrections_file, args.output_image, args.overwrite)
    except ValueError as e:
        log = logging.getLogger('rapthor:correct_astrometry')
        log.critical(e)
