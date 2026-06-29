"""Skymodel filtering helpers for image execution."""

import json
import os
import shutil
from typing import Optional, Sequence, Union

import numpy as np
from astropy.table import Table
from astropy.utils import iers
from lsmtool.filter_skymodel import filter_skymodel as lsmtool_filter_skymodel

# Avoid Astropy IERS downloads on compute nodes without internet access.
iers.conf.auto_download = False

RmsBox = Union[str, tuple[float, float], tuple[int, int]]


def filter_image_skymodel(
    flat_noise_image: str,
    true_sky_image: str,
    true_sky_skymodel: str,
    apparent_sky_skymodel: str,
    output_root: str,
    vertices_file: str,
    beam_ms: Sequence[str],
    bright_true_sky_skymodel: Optional[str] = None,
    threshisl: float = 5.0,
    threshpix: float = 7.5,
    rmsbox: RmsBox = (150, 50),
    rmsbox_bright: RmsBox = (35, 7),
    adaptive_thresh: float = 75.0,
    filter_by_mask: bool = True,
    ncores: int = 8,
    source_finder: str = "bdsf",
) -> None:
    """
    Filter apparent and true sky models using image-source detection products.

    If the source finder reports that all image pixels are blanked, valid empty
    skymodel, catalog, RMS-image, and diagnostics outputs are written instead.
    """
    true_sky_skymodel = true_sky_skymodel if os.path.exists(true_sky_skymodel) else None
    apparent_sky_skymodel = apparent_sky_skymodel if os.path.exists(apparent_sky_skymodel) else None

    try:
        nsources = lsmtool_filter_skymodel(
            flat_noise_image,
            true_sky_image,
            true_sky_skymodel,
            apparent_sky_skymodel,
            beam_ms=beam_ms,
            vertices_file=vertices_file,
            input_bright_skymodel=bright_true_sky_skymodel,
            output_apparent_sky=f"{output_root}.apparent_sky.txt",
            output_true_sky=f"{output_root}.true_sky.txt",
            output_flat_noise_rms=f"{output_root}.flat_noise_rms.fits",
            output_true_rms=f"{output_root}.true_sky_rms.fits",
            output_catalog=f"{output_root}.source_catalog.fits",
            source_finder=source_finder,
            thresh_isl=threshisl,
            thresh_pix=threshpix,
            rmsbox=rmsbox,
            rmsbox_bright=rmsbox_bright,
            adaptive_thresh=adaptive_thresh,
            filter_by_mask=filter_by_mask,
            keep_mask=True,
            ncores=ncores,
        )
    except RuntimeError as error:
        if "All pixels in the image are blanked" not in str(error):
            raise
        nsources = 0
        _write_blank_filter_outputs(
            flat_noise_image,
            true_sky_image,
            true_sky_skymodel,
            apparent_sky_skymodel,
            output_root,
        )

    _write_image_diagnostics(output_root, nsources)


def _write_empty_skymodel(input_skymodel: Optional[str], output_skymodel: str) -> None:
    """Write an empty makesourcedb skymodel, preserving the input FORMAT line."""
    header = "FORMAT = Name, Type, Ra, Dec, I"
    if input_skymodel is not None and os.path.exists(input_skymodel):
        with open(input_skymodel) as handle:
            first_line = handle.readline().strip()
        if first_line.startswith("FORMAT"):
            header = first_line
    with open(output_skymodel, "w") as handle:
        handle.write(f"{header}\n")


def _write_empty_source_catalog(output_catalog: str) -> None:
    """Write an empty FITS source catalog with the columns used downstream."""
    columns = {
        "Source_id": np.array([], dtype="<U1"),
        "RA": np.array([], dtype=float),
        "DEC": np.array([], dtype=float),
        "Isl_Total_flux": np.array([], dtype=float),
        "Total_flux": np.array([], dtype=float),
        "DC_Maj": np.array([], dtype=float),
        "E_RA": np.array([], dtype=float),
        "E_DEC": np.array([], dtype=float),
    }
    Table(columns).write(output_catalog, format="fits", overwrite=True)


def _write_blank_filter_outputs(
    flat_noise_image: str,
    true_sky_image: str,
    true_sky_skymodel: Optional[str],
    apparent_sky_skymodel: Optional[str],
    output_root: str,
) -> None:
    """Write valid empty products for an all-blank source-detection image."""
    _write_empty_skymodel(true_sky_skymodel, f"{output_root}.true_sky.txt")
    _write_empty_skymodel(apparent_sky_skymodel, f"{output_root}.apparent_sky.txt")
    shutil.copyfile(flat_noise_image, f"{output_root}.flat_noise_rms.fits")
    shutil.copyfile(true_sky_image, f"{output_root}.true_sky_rms.fits")
    _write_empty_source_catalog(f"{output_root}.source_catalog.fits")


def _write_image_diagnostics(output_root: str, nsources: int) -> None:
    """Write the number of detected sources for later diagnostics."""
    with open(f"{output_root}.image_diagnostics.json", "w") as output_file:
        json.dump({"nsources": nsources}, output_file)
