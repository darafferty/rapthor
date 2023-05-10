# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

from numpy.lib.function_base import diff
import pytest
import numpy as np
import casacore.tables
import everybeam as eb
import os

import idg
import idg.util
from idg.idgcalutils import next_composite
import astropy.io.fits as fits
from astropy import units
from utils import preparetestset, read_fits_parameters, init_buffers, init_tapered_grid


# Extract some environment variables
DATADIR = os.environ["DATADIR"]
COMMONDIR = os.environ["COMMON"]
MSNAME = os.environ["MSNAME"]
MS = os.path.join(DATADIR, MSNAME)

INTERVALSTART = 0
INTERVALEND = 100
STARTCHAN = 0
NCHAN = 0

CELLSIZE = (2 * units.arcsec).to(units.rad).value  # 2 asecs
IMAGESIZE = 512  # pixels
PADDING = 1.2
SUBGRID_SIZE = 32
KERNEL_SIZE = 16
WSTEP = 400
NX = NY = -128

STOKES = "I"
PHASE_CENTRE_RA = 2.15374
PHASE_CENTRE_DEC = 0.841552


def get_everybeam_aterms(
    ms,
    differential_beam,
    subgrid_size,
    time,
    frequency,
    phase_centre_ra,
    phase_centre_dec,
    dl,
    dm,
):
    gs = eb.GridSettings()
    gs.width = gs.height = subgrid_size
    gs.ra = phase_centre_ra
    gs.dec = phase_centre_dec
    gs.dl = dl
    gs.dm = dm
    gs.l_shift = -0.5 * gs.dl
    gs.m_shift = 0.5 * gs.dm

    telescope = eb.load_telescope(ms, use_differential_beam=differential_beam)
    return telescope.gridded_response(gs, time, frequency)


def run_degridding(
    proxy,
    ms,
    imagename,
    interval_start,
    interval_end,
    wstep,
    subgrid_size,
    kernel_size,
    padding,
    grid_with_beam,
    differential_beam,
    phase_centre_ra,
    phase_centre_dec,
    nr_timeslots=1,
    nr_correlations=4,
):
    suffix = "model-pb.fits" if grid_with_beam else "model.fits"
    image_path = imagename + "-I-" + suffix

    fits_settings = read_fits_parameters(image_path)
    cell_size = fits_settings["cell_size"]

    datacolumn = "DATA"

    # Open measurementset and read parameters
    table = casacore.tables.taql(f"SELECT * from {ms} WHERE ANTENNA1 != ANTENNA2")
    t_ant = casacore.tables.table(table.getkeyword("ANTENNA"))
    t_spw = casacore.tables.table(table.getkeyword("SPECTRAL_WINDOW"))
    frequencies = np.asarray(t_spw[0]["CHAN_FREQ"], dtype=np.float32)

    nr_stations = len(t_ant)
    # Number of baselines without auto-correlation
    nr_baselines = (nr_stations * (nr_stations - 1)) // 2
    nr_channels = table[0][datacolumn].shape[0]

    nr_timesteps = interval_end - interval_start

    # Initialize empty buffers
    rowid = np.zeros((nr_baselines, nr_timesteps), dtype=np.int)
    (uvw, visibilities, _) = init_buffers(
        nr_baselines, nr_channels, nr_timesteps, nr_correlations
    )
    baselines = np.zeros(nr_baselines, dtype=np.int32)

    # Initialize taper and grid
    d = fits.getdata(image_path)[0, 0]

    taper, grid = init_tapered_grid(
        subgrid_size, 7, padding, nr_correlations, N0=fits_settings["N0"], d=d
    )

    proxy.set_grid(grid)
    proxy.transform(idg.ImageDomainToFourierDomain)

    start_row = nr_baselines * interval_start
    nr_rows = nr_baselines * nr_timesteps

    # Read nr_timesteps samples for baselines
    antenna1_block = table.getcol("ANTENNA1", startrow=start_row, nrow=nr_rows)
    antenna2_block = table.getcol("ANTENNA2", startrow=start_row, nrow=nr_rows)
    uvw_block = table.getcol("UVW", startrow=start_row, nrow=nr_rows)
    rowid = np.arange(nr_rows).reshape(nr_baselines, nr_timesteps, order="F").flatten()

    uvw_block[:, 1:3] = -uvw_block[:, 1:3].astype(np.float32)

    # Reshape data
    antenna1_block = antenna1_block.reshape(nr_timesteps, nr_baselines)
    antenna2_block = antenna2_block.reshape(nr_timesteps, nr_baselines)
    uvw_block = uvw_block.reshape(nr_timesteps, nr_baselines, 3)

    # Take transpose
    baselines = np.array(
        [(a1, a2) for a1, a2 in zip(antenna1_block[0, :], antenna2_block[0, :])],
        dtype=np.int32,
    )

    uvw[:] = uvw_block.transpose((1, 0, 2))

    aterm_offsets = idg.util.get_example_aterm_offsets(nr_timeslots, nr_timesteps)
    if grid_with_beam:
        t_table = casacore.tables.taql(f"SELECT UNIQUE TIME FROM {ms}")
        t_centroid = np.mean(
            t_table.getcol("TIME", startrow=interval_start, nrow=nr_timesteps)
        )
        dl = dm = -cell_size * grid.shape[-1] / subgrid_size
        aterms = get_everybeam_aterms(
            ms,
            differential_beam,
            subgrid_size,
            t_centroid,
            frequencies[0],
            phase_centre_ra,
            phase_centre_dec,
            dl,
            dm,
        )
        aterms = aterms.reshape(
            1, nr_stations, subgrid_size, subgrid_size, nr_correlations
        )
    else:
        aterms = idg.util.get_identity_aterms(
            nr_timeslots, nr_stations, subgrid_size, nr_correlations
        )

    shift = np.zeros(2, np.float32)
    proxy.init_cache(subgrid_size, cell_size, wstep, shift)

    proxy.degridding(
        kernel_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        aterms,
        aterm_offsets,
        taper,
    )

    predicted_visibilities = np.zeros(
        (nr_timesteps * nr_baselines, nr_channels, nr_correlations), dtype=np.complex64,
    )
    # predicted_visibilities[rowid.flatten(), ...] = visibilities.reshape(
    predicted_visibilities[rowid, ...] = visibilities.reshape(
        (nr_timesteps * nr_baselines, nr_channels, nr_correlations)
    )
    return predicted_visibilities


def run_gridding(
    proxy,
    ms,
    interval_start,
    interval_end,
    wstep,
    imagesize,
    subgrid_size,
    kernel_size,
    padding,
    cell_size,
    grid_with_beam,
    differential_beam,
    phase_centre_ra,
    phase_centre_dec,
    nr_timeslots=1,
    nr_correlations=4,
):

    table = casacore.tables.taql(f"SELECT * from {ms} WHERE ANTENNA1 != ANTENNA2")
    datacolumn = "DATA"

    # Read parameters from measurementset
    t_ant = casacore.tables.table(table.getkeyword("ANTENNA"))
    t_spw = casacore.tables.table(table.getkeyword("SPECTRAL_WINDOW"))
    frequencies = np.asarray(t_spw[0]["CHAN_FREQ"], dtype=np.float32)
    nr_channels = table[0][datacolumn].shape[0]

    nr_stations = len(t_ant)
    nr_baselines = (nr_stations * (nr_stations - 1)) // 2

    nr_timesteps = interval_end - interval_start

    grid_size = next_composite(int(imagesize * padding))

    grid = idg.util.get_example_grid(nr_correlations, grid_size)
    aterm_offsets = idg.util.get_example_aterm_offsets(nr_timeslots, nr_timesteps)

    if grid_with_beam:
        t_table = casacore.tables.taql(f"SELECT UNIQUE TIME FROM {ms}")
        t_centroid = np.mean(
            t_table.getcol("TIME", startrow=INTERVALSTART, nrow=nr_timesteps)
        )
        dl = dm = -cell_size * grid.shape[-1] / subgrid_size
        aterms = get_everybeam_aterms(
            ms,
            differential_beam,
            subgrid_size,
            t_centroid,
            frequencies[0],
            phase_centre_ra,
            phase_centre_dec,
            dl,
            dm,
        )
        aterms = aterms.reshape(
            1, nr_stations, subgrid_size, subgrid_size, nr_correlations
        )
    else:
        aterms = idg.util.get_identity_aterms(
            nr_timeslots, nr_stations, subgrid_size, nr_correlations
        )

    # Initialize taper
    taper = idg.util.get_example_taper(subgrid_size)
    taper_grid = idg.util.get_example_taper(grid_size)

    shift = np.zeros(2, np.float32)
    proxy.set_grid(grid)
    proxy.init_cache(subgrid_size, cell_size, wstep, shift)

    start_row = nr_baselines * INTERVALSTART
    nr_rows = nr_baselines * nr_timesteps

    # Initialize empty buffers
    (uvw, visibilities, _) = init_buffers(
        nr_baselines, nr_channels, nr_timesteps, nr_correlations
    )
    baselines = np.zeros(shape=(nr_baselines), dtype=idg.baselinetype)

    # Read nr_timesteps samples for all baselines including auto correlations
    timestamp_block = table.getcol("TIME", startrow=start_row, nrow=nr_rows)
    antenna1_block = table.getcol("ANTENNA1", startrow=start_row, nrow=nr_rows)
    antenna2_block = table.getcol("ANTENNA2", startrow=start_row, nrow=nr_rows)
    uvw_block = table.getcol("UVW", startrow=start_row, nrow=nr_rows)
    vis_block = table.getcol(datacolumn, startrow=start_row, nrow=nr_rows)
    weight_block = table.getcol("WEIGHT_SPECTRUM", startrow=start_row, nrow=nr_rows)
    vis_block *= weight_block
    flags_block = table.getcol("FLAG", startrow=start_row, nrow=nr_rows)
    vis_block = vis_block * ~flags_block
    vis_block[np.isnan(vis_block)] = 0

    # Change precision
    uvw_block[:, 1:3] = -uvw_block[:, 1:3].astype(np.float32)
    vis_block = vis_block.astype(np.complex64)

    # Reshape data
    antenna1_block = antenna1_block.reshape(nr_timesteps, nr_baselines)
    antenna2_block = antenna2_block.reshape(nr_timesteps, nr_baselines)
    uvw_block = uvw_block.reshape(nr_timesteps, nr_baselines, 3)
    vis_block = vis_block.reshape(
        nr_timesteps, nr_baselines, nr_channels, nr_correlations
    )

    # Take transpose
    baselines = np.array(
        [(a1, a2) for a1, a2 in zip(antenna1_block[0, :], antenna2_block[0, :])],
        dtype=np.int32,
    )

    visibilities[...] = vis_block.transpose((1, 0, 2, 3))
    uvw[:] = uvw_block.transpose((1, 0, 2))

    # Grid visibilities
    proxy.gridding(
        kernel_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        aterms,
        aterm_offsets,
        taper,
    )

    # Copy grid in UV domain
    uv_grid = grid.copy()

    # Transform grid to image domain
    proxy.transform(idg.FourierDomainToImageDomain)
    # Get real part, and remove taper
    img_real = np.real(grid[0, ...]) / taper_grid

    clip = (grid_size - imagesize) // 2

    # Crop image and normalize on peak value
    # TODO: check normalization procedure
    img_crop = img_real[clip:-clip, clip:-clip]
    img_crop /= np.amax(img_crop)

    return timestamp_block, uv_grid, img_crop


@pytest.mark.parametrize(
    "proxy", ["idg.CPU.Optimized", "idg.HybridCUDA.GenericOptimized"]
)
@pytest.mark.parametrize("grid_with_beam", [False, True])
def test_degridding(proxy, grid_with_beam, differential_beam=True):
    try:
        proxy = eval(proxy)()
    except:
        pytest.skip(f"could not instantiate proxy " + proxy)

    # Write sourcedb and run DPPP predict
    preparetestset(
        STOKES,
        NX,
        NY,
        grid_with_beam,
        differential_beam,
        DATADIR,
        COMMONDIR,
        MS,
        INTERVALSTART,
        INTERVALEND,
        STARTCHAN,
        NCHAN,
    )

    image = f"pointsource-{STOKES}"
    visibilities = run_degridding(
        proxy,
        MS,
        image,
        INTERVALSTART,
        INTERVALEND,
        WSTEP,
        SUBGRID_SIZE,
        KERNEL_SIZE,
        PADDING,
        grid_with_beam,
        differential_beam,
        PHASE_CENTRE_RA,
        PHASE_CENTRE_DEC,
    )

    T = casacore.tables.taql(
        f"SELECT TIME, cdatetime(TIME-.1) AS TIMESTR FROM {MS} GROUPBY TIME"
    )
    starttime = T[INTERVALSTART]["TIME"] - 0.1
    endtime = T[INTERVALEND]["TIME"] - 0.1
    t = casacore.tables.taql(
        f"SELECT * FROM {MS} WHERE TIME>{starttime} AND TIME<{endtime} AND ANTENNA1 != ANTENNA2"
    )
    data = t.getcol("DATA")  # generated by DPPP
    if NCHAN > 0:
        data = data[:, STARTCHAN : STARTCHAN + NCHAN, :]

    assert np.allclose(data, visibilities, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "proxy", ["idg.CPU.Optimized", "idg.HybridCUDA.GenericOptimized"]
)
@pytest.mark.parametrize("grid_with_beam", [False, True])
def test_gridding(proxy, grid_with_beam, differential_beam=True, plot=False):
    try:
        proxy = eval(proxy)()
    except:
        pytest.skip(f"could not instantiate proxy " + proxy)

    # Run DPPP predict
    preparetestset(
        STOKES,
        NX,
        NY,
        grid_with_beam,
        differential_beam,
        DATADIR,
        COMMONDIR,
        MS,
        INTERVALSTART,
        INTERVALEND,
        STARTCHAN,
        NCHAN,
    )

    timestamp_block, uv_grid, img_crop = run_gridding(
        proxy,
        MS,
        INTERVALSTART,
        INTERVALEND,
        WSTEP,
        IMAGESIZE,
        SUBGRID_SIZE,
        KERNEL_SIZE,
        PADDING,
        CELLSIZE,
        grid_with_beam,
        differential_beam,
        PHASE_CENTRE_RA,
        PHASE_CENTRE_DEC,
    )

    # This test assumes test_gridding.py was run before: It compares the result
    # against a Q-I image, generated in test_gridding.py.
    if grid_with_beam:
        # TODO: check this. Beam doesn't seem to make much difference right now.
        wsclean_fits = fits.getdata(
            os.path.join(os.environ["WORKDIR"], "pointsource-Q-beam-I-image.fits")
        )
    else:
        wsclean_fits = fits.getdata(
            os.path.join(os.environ["WORKDIR"], "pointsource-Q-I-image.fits")
        )

    np.testing.assert_allclose(img_crop, wsclean_fits[0, 0, ...], atol=5e-2)

    if plot:
        # Plot of UV grid, image domain grid, and reference image
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(1, 3, figsize=(20, 10))

        # Set plot properties
        cmap = plt.get_cmap("hot")
        font_size = 16
        time1 = timestamp_block[0]

        # Plot grid in UV domain
        axes[0].imshow(np.log(np.abs(uv_grid[0, ...]) + 1), cmap=cmap)
        axes[0].set_title(
            f"UV Data: {np.mod(int(time1 / 3600), 24):2d}:{np.mod(int(time1 / 60), 60):2d}",
            fontsize=font_size,
        )

        # Plot processed grid in image domain
        axes[1].imshow(img_crop, interpolation="nearest", cmap=cmap)
        axes[1].set_title("Sky image\n", fontsize=font_size)

        # Plot reference (wsclean) image
        axes[2].imshow(wsclean_fits[0, 0, ...], interpolation="nearest", cmap=cmap)
        axes[2].set_title("WSClean\n", fontsize=font_size)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show(block=True)
