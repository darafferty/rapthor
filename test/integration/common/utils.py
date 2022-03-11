# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later3

import os
import numpy as np
from subprocess import check_call

import idg
from idg.idgcalutils import next_composite, idgwindow
import casacore.tables
from astropy.io import fits

"""Collection of utility functions for running the IDG tests
"""


def preparetestset(
    stokes,
    nx,
    ny,
    grid_with_beam,
    differential_beam,
    datadir,
    commondir,
    ms,
    intervalstart,
    intervalend,
    start_channel,
    nchannels,
):
    """
    Prepare the test measurementset by predicting DATA column with DP3

    TODO: this function is now called both by test_gridding as test_degridding.
    Maybe this can be done a bit more efficient
    """
    offset = (int(ny), int(nx))
    template = os.path.join(datadir, "template.fits")
    suffix = "model-pb.fits" if grid_with_beam else "model.fits"

    stokes1 = stokes
    with fits.open(template) as img:
        N = img[0].data.shape[-1]
        img[0].data[:] = 0.0

        for stokes2 in ["Q", "U", "V"]:
            if stokes1 != stokes2:
                img.writeto(
                    f"pointsource-{stokes1}-{stokes2}-{suffix}", overwrite=True,
                )

        img[0].data[0, 0, int(N / 2 + offset[0]), int(N / 2 + offset[1])] = 1.0

        img.writeto(f"pointsource-{stokes1}-I-{suffix}", overwrite=True)
        if stokes1 != "I":
            img.writeto(f"pointsource-{stokes1}-{stokes1}-{suffix}", overwrite=True)

    check_call(
        [
            "casapy2bbs.py",
            "--no-patches",
            f"pointsource-{stokes1}-{stokes1}-{suffix}",
            "temp.cat",
        ]
    )

    full_stokes_list = ["I", "Q", "U", "V"]
    stokesI_idx = 4
    stokes_idx = 4 + full_stokes_list.index(stokes)

    file_in = "temp.cat"
    file_out = f"pointsource-{stokes}.cat"

    with open(file_in, "r") as f_in:
        contents = f_in.readlines()

    # Extract and modify line 7 from the temp.cat file
    patch_info = contents[6].split(",")
    patch_info[stokes_idx] = patch_info[stokesI_idx]
    contents[6] = ",".join(patch_info)

    # Write cat file for corresponding component
    file_out = f"pointsource-{stokes}.cat"
    with open(file_out, "w") as f_out:
        f_out.writelines(contents)

    sourcedb = f"pointsource-{stokes}.sourcedb"
    check_call(["rm", "-rf", sourcedb])
    check_call(
        [
            "makesourcedb",
            f"in={file_out}",
            "format=Name, Type, Ra, Dec, I, Q, U, V",
            f"out={sourcedb}",
        ]
    )

    T = casacore.tables.taql(
        "SELECT TIME, cdatetime(TIME-.1) AS TIMESTR FROM $ms GROUPBY TIME"
    )

    starttimestr = T[intervalstart]["TIMESTR"]
    endtimestr = T[intervalend]["TIMESTR"]

    check_call(
        [
            "DP3",
            os.path.join(
                commondir,
                (
                    "dp3-predict-correct.parset"
                    if (grid_with_beam and differential_beam)
                    else "dp3-predict.parset"
                ),
            ),
            f"msin={ms}",
            f"msin.starttime={starttimestr}",
            f"msin.endtime={endtimestr}",
            f"msin.startchan={start_channel}",
            f"msin.nchan={nchannels}",
            f"predict.sourcedb={sourcedb}",
            f"predict.usebeammodel={grid_with_beam}",
        ]
    )


def read_fits_parameters(fits_path):
    """
    Read parameters from a provided fits file and set
    padding properties

    Parameters
    ----------
    fits_path : str

    Returns
    -------
    dict
        Dictionary setting containing some properties of
        the fits file
    """
    fits_dict = {}
    h = fits.getheader(fits_path)

    fits_dict["cell_size"] = abs(h["CDELT1"]) / 180 * np.pi
    fits_dict["N0"] = h["NAXIS1"]
    return fits_dict


def init_buffers(nr_baselines, nr_channels, nr_timesteps, nr_correlations):
    # Initialize empty buffers
    uvw = np.zeros(shape=(nr_baselines, nr_timesteps, 3), dtype=np.float32)
    visibilities = np.zeros(
        shape=(nr_baselines, nr_timesteps, nr_channels, nr_correlations),
        dtype=idg.visibilitiestype,
    )
    weights = np.zeros(
        shape=(nr_baselines, nr_timesteps, nr_channels, nr_correlations),
        dtype=np.float32,
    )
    return (uvw, visibilities, weights)


def init_tapered_grid(subgrid_size, taper_support, padding, nr_correlations, N0, d):
    """
    Initialize tapered grid

    Parameters
    ----------
    subgrid_size : int
        Subgrid size
    taper_support : int
        Taper support
    padding : float
        (Sub)grid padding
    nr_correlations : int
        Number of correlations
    N0 : int
        (unpadded) image size
    d : np.2darray
        2D array of image

    Returns
    -------
    taper, grid
    """

    assert d.shape == (N0, N0)

    # Initialize taper
    taper = idgwindow(subgrid_size, taper_support, padding)

    # Compute grid
    N = next_composite(int(N0 * padding))
    grid_size = N

    taper_ = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(taper)))
    taper_grid = np.zeros(grid_size, dtype=np.complex128)
    taper_grid[(grid_size - subgrid_size) // 2 : (grid_size + subgrid_size) // 2] = (
        taper_
        * np.exp(-1j * np.linspace(-np.pi / 2, np.pi / 2, subgrid_size, endpoint=False))
    )
    taper_grid = (
        np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(taper_grid))).real
        * grid_size
        / subgrid_size
    )
    taper_grid0 = taper_grid[(N - N0) // 2 : (N + N0) // 2]

    grid = np.zeros(shape=(nr_correlations, grid_size, grid_size), dtype=idg.gridtype)
    grid[
        0, (N - N0) // 2 : (N + N0) // 2, (N - N0) // 2 : (N + N0) // 2
    ] = d / np.outer(taper_grid0, taper_grid0)
    grid[
        3, (N - N0) // 2 : (N + N0) // 2, (N - N0) // 2 : (N + N0) // 2
    ] = d / np.outer(taper_grid0, taper_grid0)
    taper = np.outer(taper, taper).astype(np.float32)
    return taper, grid
