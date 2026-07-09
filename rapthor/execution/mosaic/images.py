"""FITS image helpers used by mosaic execution scripts."""

import os
import re
import shutil
from typing import Sequence

import numpy as np
from astropy.io import fits as pyfits
from astropy.wcs import WCS as pywcs
from lsmtool.io import read_vertices_x_y
from reproject import reproject_interp
from shapely import contains_xy as polygon_contains_xy
from shapely.geometry import Polygon

from rapthor.lib import miscellaneous as misc
from rapthor.lib.fitsimage import FITSImage

_SPARSE_MODEL_PRODUCT = re.compile(r"(^|[-_])(?:filtered[-_])?model([-_.]|$)")


def is_sparse_model_product(filename: str) -> bool:
    """Return true for sparse WSClean model-image mosaic products."""
    return bool(_SPARSE_MODEL_PRODUCT.search(os.path.basename(filename).lower()))


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


def regrid_sparse_model_image(
    input_image: str,
    template_image: str,
    vertices_file: str,
    output_image: str,
    *,
    skip: bool = False,
) -> None:
    """
    Regrid a sparse model image without interpolating clean components.

    Model images contain sparse clean-component pixels, not a continuous sky
    brightness image. Interpolating them can spread flux into artificial
    stripe-like structures, so nonzero finite pixels are mapped to the nearest
    output pixel while the valid sector footprint stays zero-valued.
    """
    if skip:
        _copy_file(input_image, output_image)
        return

    with pyfits.open(template_image) as hdul:
        regrid_header = hdul[0].header.copy()
        output_data = np.full_like(hdul[0].data, np.nan, dtype=float)

    wcs_out = pywcs(regrid_header)
    output_polygon = _sector_polygon(vertices_file, wcs_out)
    _fill_polygon_footprint(output_data, output_polygon)

    image = FITSImage(input_image)
    source_polygon = _sector_polygon(vertices_file, image.get_wcs())
    source_mask = np.isfinite(image.img_data) & (image.img_data != 0)
    if np.any(source_mask):
        source_y, source_x = np.nonzero(source_mask)
        in_source_footprint = polygon_contains_xy(source_polygon, source_x, source_y)
        source_x = source_x[in_source_footprint]
        source_y = source_y[in_source_footprint]
        target_x_float, target_y_float = wcs_out.world_to_pixel(
            image.get_wcs().pixel_to_world(source_x, source_y)
        )
        valid_projection = np.isfinite(target_x_float) & np.isfinite(target_y_float)
        target_x = np.rint(target_x_float[valid_projection]).astype(int)
        target_y = np.rint(target_y_float[valid_projection]).astype(int)
        source_values = image.img_data[source_y[valid_projection], source_x[valid_projection]]

        in_bounds = (
            (target_x >= 0)
            & (target_y >= 0)
            & (target_y < output_data.shape[0])
            & (target_x < output_data.shape[1])
        )
        target_x = target_x[in_bounds]
        target_y = target_y[in_bounds]
        source_values = source_values[in_bounds]
        in_footprint = np.isfinite(output_data[target_y, target_x])
        np.add.at(
            output_data,
            (target_y[in_footprint], target_x[in_footprint]),
            source_values[in_footprint],
        )

    pyfits.PrimaryHDU(header=regrid_header, data=output_data).writeto(output_image, overwrite=True)


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
            mosaic_slice = mosaic_data[output_slice]
            covered = weights > 0
            np.divide(mosaic_slice, weights, out=mosaic_slice, where=covered)
            mosaic_slice[~covered] = np.nan

    mosaic_data[~np.isfinite(mosaic_data)] = np.nan
    pyfits.PrimaryHDU(header=mosaic_header, data=mosaic_data).writeto(output_image, overwrite=True)


def _copy_file(input_file: str, output_file: str) -> None:
    """Copy ``input_file`` to ``output_file``, replacing an existing output."""
    if os.path.exists(output_file):
        os.remove(output_file)
    shutil.copyfile(input_file, output_file)


def _sector_polygon(vertices_file: str, wcs: pywcs) -> Polygon:
    """Return the padded sector polygon in pixel coordinates for ``wcs``."""
    vertices = read_vertices_x_y(vertices_file, wcs)
    return Polygon(vertices).buffer(2)


def _fill_polygon_footprint(data: np.ndarray, polygon: Polygon) -> None:
    """Set pixels inside ``polygon`` to zero, leaving all other pixels as NaN."""
    min_x, min_y, max_x, max_y = polygon.bounds
    x_0 = max(0, int(np.floor(min_x)))
    y_0 = max(0, int(np.floor(min_y)))
    x_1 = min(data.shape[1], int(np.ceil(max_x)) + 1)
    y_1 = min(data.shape[0], int(np.ceil(max_y)) + 1)
    if x_1 <= x_0 or y_1 <= y_0:
        return
    yy, xx = np.indices((y_1 - y_0, x_1 - x_0))
    yy += y_0
    xx += x_0
    inside = polygon_contains_xy(polygon, xx.ravel(), yy.ravel()).reshape(xx.shape)
    footprint = data[y_0:y_1, x_0:x_1]
    footprint[inside] = 0.0
