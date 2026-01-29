#!/usr/bin/env python3
"""
Script to restore a skymodel into an image
"""

from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Tuple, Union
import numpy as np
from astropy.io.fits import open as fits_open, writeto as fits_write, CompImageHDU, FitsHDU
import subprocess 
from pathlib import Path
import logging
from tempfile import NamedTemporaryFile
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def restore_with_wsclean(source_catalog: Path, reference_image: Path, beam_size: float) -> Path:
    """
    Restore a skymodel into an image using wsclean.

    Parameters
    ----------
    source_catalog : Path
        Source catalog file in text format
    reference_image : Path
        Reference image file in FITS format
    beam_size : float
        Size of the restoring beam in arcseconds.

    Returns
    -------
    Path
        Restored skymodel image path
    """
    logger.info("Restoring sky model using wsclean.")
    output_image = Path(NamedTemporaryFile(suffix=".fits", delete=False).name)
    cmd = [
        "wsclean",
        "-restore-list",
        str(reference_image),
        str(source_catalog),
        str(output_image),
        "-beam-size", str(beam_size)
    ]
    subprocess.run(cmd, check=True)
    logger.info(f"Restored image saved to {output_image}")
    return output_image


def log_fits_info(fits_obj):
    for i, name, version, type, cards, dimensions, format, _ in fits_obj.info(output=False):
        logger.info("HDU %s: name=%s, version=%s, type=%s, cards=%s, dimensions=%s, format=%s",
                    i, name, version, type, cards, dimensions, format)   


def get_primary_hdu_or_compressed(fits_obj) -> Union[FitsHDU, CompImageHDU]:
    """
    Get the primary HDU or the compressed image HDU from a FITS object.

    Parameters
    ----------
    fits_obj
        FITS object

    Returns
    -------
    FitsHDU | CompImageHDU
        The primary HDU or compressed image HDU
    """
    compressed_hdu = next(
        (hdu for hdu in fits_obj
        if isinstance(hdu, CompImageHDU)), None
    )
    if compressed_hdu:
        return compressed_hdu
    else:
        return fits_obj[0]


def derive_minimum_scale(header) -> float:
    """
    Derive the minimum scale from the FITS header.

    Parameters
    ----------
    header
        FITS header

    Returns
    -------
    float
        Minimum scale in arcseconds
    """
    min_scale = None
    for axis in range(1, header["NAXIS"] + 1):
        if "RA" in str(header[f"CTYPE{axis}"]) or "DEC" in str(header[f"CTYPE{axis}"]):
            if min_scale is None:
                min_scale = abs(float(header[f"CDELT{axis}"])) * 3600.0  # arcseconds
            else:
                min_scale = min(min_scale, abs(float(header[f"CDELT{axis}"]))) * 3600.0  # arcseconds
    if min_scale is None:
        raise ValueError("Could not determine minimum scale from header.")
    return min_scale


def make_zero_image(reference_image: Path) -> Tuple[Path, float]:
    """
    Make a zero image based on the dimensions and header of a reference image.

    Parameters
    ----------
    reference_image : Path
        Reference image path

    Returns
    -------
    Path, float
        Path to the created zero image and scale factor for pixels
    """
    with fits_open(reference_image) as ref_fits:
        logger.info("Creating zero image based on reference image dimensions and header.")
        logger.info("Reference image: %s", reference_image)

        log_fits_info(ref_fits)

        ref_image = get_primary_hdu_or_compressed(ref_fits)
        header = ref_image.header
        min_scale = derive_minimum_scale(header)
        logger.info(f"Derived minimum scale: {min_scale} arcseconds")
        data = np.zeros_like(ref_image.data)
        output_image = Path(NamedTemporaryFile(suffix=".fits", delete=False).name)
        fits_write(output_image, data=data, header=header, overwrite=True)
        logger.info(f"Zero image saved to {output_image.name}")
        return output_image, min_scale


def compress_image_if_needed(input_image: Path, output_image: Path)-> Path:
    """
    Compress the image if the output filename indicates compression.

    Parameters
    ----------
    input_image : Path
        Input image path
    output_image : Path
        Output image path

    Returns
    -------
    Path
        Path to the output image
    """
    if output_image.suffixes != ".fits":
        logger.info("Compressing output image %s", output_image)
        with fits_open(input_image) as fits_obj:
            fits_obj.writeto(output_image, overwrite=True)
        logger.info("Compressed image saved to %s", output_image)
    else:
        shutil.copy(input_image, output_image)
    return output_image


def main(source_catalog: Path, reference_image: Path, output_image: Path):
    """
    Main entrypoint for restoring a skymodel into an image.

    Parameters
    ----------
    source_catalog : Path
        Source catalog path
    reference_image : Path
        Reference image path
    output_image : Path
        Output image path
    """
    temp_images = []
    temp_image, pixel_scale = make_zero_image(reference_image)
    temp_images.append(temp_image)
    try:
        restored_image = restore_with_wsclean(source_catalog, temp_image, pixel_scale)
        temp_images.append(restored_image)
        compress_image_if_needed(restored_image, output_image)
    except Exception as e:
        logger.exception("An error occurred during the restoration process: %s", e)
    finally:
        for image in temp_images:
            if image.exists():
                image.unlink()


if __name__ == "__main__":
    descriptiontext = "Restore a skymodel text file into an image.\n" \
                      "The restored image will have the same dimensions and WCS as the reference image." \
                      

    parser = ArgumentParser(description=descriptiontext, formatter_class=RawTextHelpFormatter)
    parser.add_argument("source_catalog", help="Filename of input FITS source catalog", type=Path)
    parser.add_argument("reference_image", help="Filename of image to use as reference for restoration", type=Path)
    parser.add_argument("output_image", help="Filename of output restored image, if (fs.gz, fits.gz, etc..) extension is used" \
                                             " it will be compressed", type=Path)
    args = parser.parse_args()
    main(args.source_catalog, args.reference_image, args.output_image)