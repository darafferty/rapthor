"""Astrometry-corrected image product helpers."""

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Optional, Union

from rapthor.execution.outputs import require_file

logger = logging.getLogger("rapthor:image:astrometry")

_REQUIRED_CORRECTION_KEYS = (
    "facet_name",
    "meanRAOffsetDeg",
    "meanDecOffsetDeg",
    "stdRAOffsetDeg",
    "stdDecOffsetDeg",
)


PathInput = Union[str, Path]


def astrometry_corrected_image_path(input_image: PathInput) -> Path:
    """Return the astrometry-corrected product path for a PB-corrected image."""
    input_path = Path(input_image)
    root = input_path.name
    compressed = root.endswith(".fz")
    if compressed:
        root = root[:-3]
    if root.endswith(".fits"):
        root = root[:-5]
    output_path = input_path.with_name(f"{root}-ast.fits")
    if compressed:
        output_path = output_path.with_suffix(output_path.suffix + ".fz")
    return output_path


def validate_astrometry_corrections(corrections: Mapping[str, object]) -> Mapping[str, object]:
    """Validate the astrometry corrections structure produced by diagnostics."""
    for key in _REQUIRED_CORRECTION_KEYS:
        if key not in corrections:
            raise ValueError(f"Missing key in corrections dict: {key}")

    if not len({len(value) for value in corrections.values()}) == 1:
        raise ValueError("Corrections should have equal length")

    return corrections


def _load_corrections(corrections_file: Optional[PathInput]) -> Optional[Mapping[str, object]]:
    if corrections_file is None:
        return None

    corrections_path = Path(corrections_file)
    with corrections_path.open("r") as fp:
        corrections = json.load(fp)
    if not corrections or not corrections.get("facet_name"):
        return None
    return validate_astrometry_corrections(corrections)


def _copy_uncorrected_image(input_image: Path, output_image: Path, overwrite: bool) -> Path:
    output_image.parent.mkdir(parents=True, exist_ok=True)
    if not output_image.exists() or overwrite:
        shutil.copy(input_image, output_image)
    return output_image


def _compressed_output_path(output_image: Path) -> Path:
    if output_image.suffix == ".fz":
        return output_image
    return output_image.with_suffix(output_image.suffix + ".fz")


def correct_astrometry_image(
    input_image: PathInput,
    region_file: Optional[PathInput],
    corrections_file: Optional[PathInput],
    *,
    output_image: Optional[PathInput] = None,
    overwrite: bool = True,
) -> Path:
    """Apply facet astrometry corrections to a Stokes-I PB-corrected image.

    If no usable correction file is available, the uncorrected image is copied
    to the astrometry-corrected product path. This preserves the product
    contract used by master for runs where astrometry diagnostics are skipped.
    """
    input_path = Path(input_image)
    output_path = (
        Path(output_image)
        if output_image is not None
        else astrometry_corrected_image_path(input_path)
    )
    corrections = _load_corrections(corrections_file)
    if corrections is None:
        if input_path.suffix == ".fz":
            output_path = _compressed_output_path(output_path)
        return _copy_uncorrected_image(input_path, output_path, overwrite)

    if region_file is None:
        logger.warning(
            "Astrometry offsets are available for %s, but no facet region file was provided; "
            "copying the uncorrected image to %s",
            input_path,
            output_path,
        )
        return _copy_uncorrected_image(input_path, output_path, overwrite)

    if input_path.suffix != ".fz":
        _apply_astrometry_corrections(
            input_path,
            Path(region_file),
            corrections,
            output_path,
            overwrite,
        )
        return output_path

    compressed_output = _compressed_output_path(output_path)
    uncompressed_output = compressed_output.with_suffix("")
    _apply_astrometry_corrections(
        input_path,
        Path(region_file),
        corrections,
        uncompressed_output,
        overwrite,
    )
    try:
        subprocess.run(["fpack", str(uncompressed_output)], check=True)
    except subprocess.CalledProcessError as err:
        print(err, file=sys.stderr)
        raise
    uncompressed_output.unlink()
    return compressed_output


def make_astrometry_corrected_image_record(
    pb_image: Mapping[str, str],
    region_record: Optional[Mapping[str, str]],
    offsets_record: Optional[Mapping[str, str]],
) -> dict:
    """Create and return the Stokes-I astrometry-corrected image record."""
    region_file = None if region_record is None else region_record["path"]
    offsets_file = None if offsets_record is None else offsets_record["path"]
    output_path = correct_astrometry_image(pb_image["path"], region_file, offsets_file)
    return require_file(str(output_path), "Astrometry-corrected PB image")


def _apply_astrometry_corrections(
    input_image: Path,
    region_file: Path,
    corrections: Mapping[str, object],
    output_image: Path,
    overwrite: bool,
) -> None:
    import numpy as np
    import scipy.ndimage as nd
    from astropy.io.fits import writeto as fits_write
    from lsmtool.facet import read_ds9_region_file
    from lsmtool.utils import rasterize

    from rapthor.lib.fitsimage import FITSImage

    uncorrected_image = FITSImage(input_image)
    wcs = uncorrected_image.get_wcs()
    ra_scale = wcs.wcs.cdelt[0]
    dec_scale = wcs.wcs.cdelt[1]
    corrected_data = np.zeros_like(uncorrected_image.img_data)
    sum_map = np.zeros_like(uncorrected_image.img_data)
    facets = read_ds9_region_file(region_file, wcs=wcs)
    facet_map = {name: index for index, name in enumerate(corrections["facet_name"])}

    for index, facet in enumerate(facets):
        logger.info("Processing facet %d/%d", index, len(facets))
        poly_padded = facet.polygon.buffer(2)
        vertices = list(
            zip(
                poly_padded.exterior.coords.xy[0].tolist(),
                poly_padded.exterior.coords.xy[1].tolist(),
            )
        )
        facet_data = rasterize(vertices, uncorrected_image.img_data.copy())

        if facet.name not in corrections["facet_name"]:
            logger.warning(
                "Astrometry offsets for facet %s were not found; leaving it unshifted",
                facet.name,
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
                    "Skipping correction for facet %s since total shift "
                    "(%.3f +/- %.3f arcsec) is not significant",
                    facet.name,
                    total_correction * ra_scale * 3600,
                    total_error * dec_scale * 3600,
                )

        poly_padded = facet.polygon.buffer(1)
        vertices = list(
            zip(
                poly_padded.exterior.coords.xy[0].tolist(),
                poly_padded.exterior.coords.xy[1].tolist(),
            )
        )
        facet_mask = rasterize(vertices, np.ones_like(facet_data))
        corrected_data += facet_mask * facet_data
        sum_map += facet_mask

    mask = sum_map > 0
    corrected_data[mask] /= sum_map[mask]
    output_shape = [1 for _ in range(uncorrected_image.header["NAXIS"] - 2)] + list(
        corrected_data.shape
    )
    fits_write(
        output_image,
        data=corrected_data.reshape(output_shape),
        header=uncorrected_image.header,
        overwrite=overwrite,
    )
