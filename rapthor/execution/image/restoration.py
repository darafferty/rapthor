"""Skymodel restoration helpers for image execution."""

import logging
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple, Union

import numpy as np
from astropy.io.fits import CompImageHDU, FitsHDU
from astropy.io.fits import open as fits_open
from astropy.io.fits import writeto as fits_write

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def restore_with_wsclean(source_catalog: Path, reference_image: Path, beam_size: float) -> Path:
    """
    Restore a skymodel into an image using WSClean.

    Parameters
    ----------
    source_catalog : Path
        Source catalog file in text format.
    reference_image : Path
        Reference image file in FITS format.
    beam_size : float
        Size of the restoring beam in arcseconds.

    Returns
    -------
    Path
        Restored skymodel image path.
    """
    logger.info("Restoring sky model using wsclean.")
    output_image = Path(NamedTemporaryFile(suffix=".fits", delete=False).name)
    command = [
        "wsclean",
        "-restore-list",
        str(reference_image),
        str(source_catalog),
        str(output_image),
        "-beam-size",
        str(beam_size),
    ]
    subprocess.run(command, check=True)
    logger.info("Restored image saved to %s", output_image)
    return output_image


def log_fits_info(fits_obj) -> None:
    """Log a compact summary of the HDUs in a FITS object."""
    for index, name, version, hdu_type, cards, dimensions, format_, _ in fits_obj.info(
        output=False
    ):
        logger.info(
            "HDU %s: name=%s, version=%s, type=%s, cards=%s, dimensions=%s, format=%s",
            index,
            name,
            version,
            hdu_type,
            cards,
            dimensions,
            format_,
        )


def get_primary_hdu_or_compressed(fits_obj) -> Union[FitsHDU, CompImageHDU]:
    """Return the compressed image HDU if present, otherwise the primary HDU."""
    compressed_hdu = next((hdu for hdu in fits_obj if isinstance(hdu, CompImageHDU)), None)
    if compressed_hdu:
        return compressed_hdu
    return fits_obj[0]


def derive_minimum_scale(header) -> float:
    """Derive the minimum RA/DEC pixel scale from a FITS header in arcseconds."""
    pixel_scales = [
        abs(float(header[f"CDELT{axis}"])) * 3600.0
        for axis in range(1, header["NAXIS"] + 1)
        if "RA" in (ctype := str(header[f"CTYPE{axis}"])) or "DEC" in ctype
    ]
    if not pixel_scales:
        raise ValueError("Could not determine minimum scale from header.")
    return min(pixel_scales)


def make_zero_image(reference_image: Path) -> Tuple[Path, float]:
    """
    Make a zero-valued FITS image with the dimensions and header of a reference image.

    Returns the temporary image path and the minimum pixel scale in arcseconds.
    """
    with fits_open(reference_image) as reference_fits:
        logger.info("Creating zero image based on reference image dimensions and header.")
        logger.info("Reference image: %s", reference_image)
        log_fits_info(reference_fits)

        reference_hdu = get_primary_hdu_or_compressed(reference_fits)
        header = reference_hdu.header
        minimum_scale = derive_minimum_scale(header)
        logger.info("Derived minimum scale: %.3f arcseconds", minimum_scale)
        data = np.zeros_like(reference_hdu.data)
        output_image = Path(NamedTemporaryFile(suffix=".fits", delete=False).name)
        fits_write(output_image, data=data, header=header, overwrite=True)
        logger.info("Zero image saved to %r", output_image.name)
        return output_image, minimum_scale


def compress_image_if_needed(input_image: Path, output_image: Path) -> Path:
    """Write ``input_image`` to ``output_image``, using FITS compression if requested."""
    if output_image.suffix != ".fits":
        logger.info("Compressing output image %s", output_image)
        with fits_open(input_image) as fits_obj:
            fits_obj.writeto(output_image, overwrite=True)
        logger.info("Compressed image saved to %s", output_image)
    else:
        shutil.copy(input_image, output_image)
    return output_image


def restore_skymodel(source_catalog: Path, reference_image: Path, output_image: Path) -> None:
    """Restore a skymodel into an image matching the reference image geometry."""
    temp_image, pixel_scale = make_zero_image(reference_image)
    temp_images = [temp_image]
    try:
        restored_image = restore_with_wsclean(source_catalog, temp_image, pixel_scale)
        temp_images.append(restored_image)
        compress_image_if_needed(restored_image, output_image)
    except Exception as error:
        logger.exception("An error occurred during the restoration process: %s", error)
    finally:
        for image in temp_images:
            if image.exists():
                image.unlink()
