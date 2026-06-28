"""FITS image helpers used by mosaic execution scripts."""

import os
import shutil
from typing import Sequence

import numpy as np
from astropy.io import fits as pyfits
from astropy.wcs import WCS as pywcs
from reproject import reproject_interp

from rapthor.lib import miscellaneous as misc
from rapthor.lib.fitsimage import FITSImage


def make_mosaic_template(
    input_image_filenames: Sequence[str],
    vertices_filenames: Sequence[str],
    output_image: str,
    *,
    skip: bool = False,
    padding: float = 1.1,
) -> None:
    """
    Make a zero-valued FITS template large enough to contain all mosaic sectors.

    Each input image is blanked outside its sector polygon before the template
    extent is calculated. ``vertices_filenames`` must therefore be ordered to
    match ``input_image_filenames``.
    """
    if skip:
        return

    directions = []
    for image_file, vertices_file in zip(input_image_filenames, vertices_filenames):
        image = FITSImage(image_file)
        image.vertices_file = vertices_file
        image.blank()
        directions.append(image)

    mra = np.mean(np.array([image.get_wcs().wcs.crval[0] for image in directions]))
    mdec = np.mean(np.array([image.get_wcs().wcs.crval[1] for image in directions]))

    rwcs = pywcs(naxis=2)
    rwcs.wcs.ctype = directions[0].get_wcs().wcs.ctype
    rwcs.wcs.cdelt = directions[0].get_wcs().wcs.cdelt
    rwcs.wcs.crval = [mra, mdec]
    rwcs.wcs.crpix = [1, 1]

    xmin, xmax, ymin, ymax = 0, 0, 0, 0
    for image in directions:
        image_wcs = image.get_wcs()
        ys, xs = np.where(image.img_data)
        axmin, aymin, axmax, aymax = xs.min(), ys.min(), xs.max(), ys.max()
        for x, y in ((axmin, aymin), (axmax, aymin), (axmin, aymax), (axmax, aymax)):
            ra, dec = map(float, image_wcs.wcs_pix2world(x, y, misc.WCS_ORIGIN))
            nx, ny = map(float, rwcs.wcs_world2pix(ra, dec, misc.WCS_ORIGIN))
            xmin, xmax, ymin, ymax = min(nx, xmin), max(nx, xmax), min(ny, ymin), max(ny, ymax)

    xsize = int(xmax - xmin)
    ysize = int(ymax - ymin)
    xpad = int(xsize * (padding - 1.0) / 2.0)
    ypad = int(ysize * (padding - 1.0) / 2.0)
    xmax += xpad
    xmin -= xpad
    ymax += ypad
    ymin -= ypad
    xsize = int(xmax - xmin)
    ysize = int(ymax - ymin)

    rwcs.wcs.crpix = [-int(xmin) + 1, -int(ymin) + 1]
    regrid_header = rwcs.to_header()
    regrid_header["NAXIS"] = 2
    regrid_header["NAXIS1"] = xsize
    regrid_header["NAXIS2"] = ysize
    for key in ("BMAJ", "BMIN", "BPA", "FREQ", "RESTFREQ", "EQUINOX"):
        if key in directions[0].img_hdr:
            regrid_header[key] = directions[0].img_hdr[key]
    regrid_header["ORIGIN"] = "Raptor"
    regrid_header["UNITS"] = "Jy/beam"
    regrid_header["TELESCOP"] = "LOFAR"

    data = np.zeros([ysize, xsize])
    pyfits.PrimaryHDU(header=regrid_header, data=data).writeto(output_image, overwrite=True)


def regrid_image(
    input_image: str,
    template_image: str,
    vertices_file: str,
    output_image: str,
    *,
    skip: bool = False,
) -> None:
    """
    Regrid one sector image to a mosaic template and blank outside its polygon.

    When ``skip`` is true, the input image is copied unchanged to
    ``output_image``.
    """
    if skip:
        _copy_file(input_image, output_image)
        return

    with pyfits.open(template_image) as hdul:
        regrid_header = hdul[0].header.copy()
        output_data = hdul[0].data.copy()
    output_data[:] = np.nan
    shape_out = output_data.shape
    wcs_out = pywcs(regrid_header)

    image = FITSImage(input_image)
    image.vertices_file = vertices_file
    image.blank()
    wcs_in = image.get_wcs()

    ny, nx = image.img_data.shape
    xc = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
    yc = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
    xc_out, yc_out = wcs_out.world_to_pixel(wcs_in.pixel_to_world(xc, yc))
    imin = max(0, int(np.floor(xc_out.min() + 0.5)))
    imax = min(shape_out[1], int(np.ceil(xc_out.max() + 0.5)))
    jmin = max(0, int(np.floor(yc_out.min() + 0.5)))
    jmax = min(shape_out[0], int(np.ceil(yc_out.max() + 0.5)))

    wcs_out_indiv = wcs_out.deepcopy()
    wcs_out_indiv.wcs.crpix[0] -= imin
    wcs_out_indiv.wcs.crpix[1] -= jmin
    shape_out_indiv = (jmax - jmin, imax - imin)

    output_slice = slice(jmin, jmax), slice(imin, imax)
    output_data[output_slice] = reproject_interp(
        (image.img_data, wcs_in),
        output_projection=wcs_out_indiv,
        shape_out=shape_out_indiv,
        return_footprint=False,
    )
    image.img_data = output_data
    image.img_hdr = regrid_header
    image.write(output_image)


def make_mosaic(
    input_image_filenames: Sequence[str],
    template_image: str,
    output_image: str,
    *,
    skip: bool = False,
) -> None:
    """
    Average finite regridded sector images into one mosaic FITS image.

    When ``skip`` is true, the first input image is copied unchanged to
    ``output_image``.
    """
    if skip:
        _copy_file(input_image_filenames[0], output_image)
        return

    with pyfits.open(template_image) as hdul:
        mosaic_header = hdul[0].header.copy()
        mosaic_data = hdul[0].data.copy()

    x_slices = [
        slice(0, int(mosaic_data.shape[0] / 2.0)),
        slice(int(mosaic_data.shape[0] / 2.0), mosaic_data.shape[0]),
    ]
    y_slices = [
        slice(0, int(mosaic_data.shape[1] / 2.0)),
        slice(int(mosaic_data.shape[1] / 2.0), mosaic_data.shape[1]),
    ]
    for x_slice in x_slices:
        for y_slice in y_slices:
            output_slice = x_slice, y_slice
            weights = np.zeros_like(mosaic_data[output_slice])
            for sector_image in input_image_filenames:
                with pyfits.open(sector_image) as hdul:
                    sector_data = hdul[0].section[output_slice].copy()
                sector_weights = np.ones_like(sector_data)
                sector_weights[~np.isfinite(sector_data)] = 0
                sector_data[~np.isfinite(sector_data)] = 0
                mosaic_data[output_slice] += sector_data
                weights += sector_weights
            mosaic_data[output_slice] /= weights

    mosaic_data[~np.isfinite(mosaic_data)] = np.nan
    pyfits.PrimaryHDU(header=mosaic_header, data=mosaic_data).writeto(output_image, overwrite=True)


def _copy_file(input_file: str, output_file: str) -> None:
    """Copy ``input_file`` to ``output_file``, replacing an existing output."""
    if os.path.exists(output_file):
        os.remove(output_file)
    shutil.copyfile(input_file, output_file)
