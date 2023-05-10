#!/usr/bin/env python
# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
from matplotlib.pyplot import grid
import idg
import idg.util as util
from idg.data import Data
import numpy as np

NR_CHANNELS = 1
NR_TIMESTEPS = 1 * 60 * 60
NR_TIMESLOTS = 16
SUBGRID_SIZE = 32
GRID_SIZE = 256
INTEGRATION_TIME = 0.9
KERNEL_SIZE = 9
NR_CORRELATIONS = 4
LAYOUT_FILE = "SKA1_low.txt"
NR_STATIONS = 12


def plot_metadata(
    kernel_size,
    subgrid_size,
    grid_size,
    cell_size,
    image_size,
    frequencies,
    uvw,
    baselines,
    aterm_offsets,
    max_nr_timesteps=np.iinfo(np.int32).max,
):
    # Debugging only
    plan = idg.Plan(
        kernel_size,
        subgrid_size,
        grid_size,
        cell_size,
        frequencies,
        uvw,
        baselines,
        aterm_offsets,
        max_nr_timesteps,
    )
    nr_subgrids = plan.get_nr_subgrids()
    metadata = np.zeros(nr_subgrids, dtype=idg.metadatatype)
    plan.copy_metadata(metadata)
    util.plot_metadata(metadata, uvw, frequencies, grid_size, subgrid_size, image_size)


def gridding(
    p,
    w_step,
    shift,
    cell_size,
    kernel_size,
    subgrid_size,
    frequencies,
    visibilities,
    uvw,
    baselines,
    aterms,
    aterm_offsets,
    taper,
):
    p.init_cache(subgrid_size, cell_size, w_step, shift)
    p.gridding(
        kernel_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        aterms,
        aterm_offsets,
        taper,
    )
    p.get_final_grid()


def degridding(
    p,
    w_step,
    shift,
    cell_size,
    kernel_size,
    subgrid_size,
    frequencies,
    visibilities,
    uvw,
    baselines,
    aterms,
    aterm_offsets,
    taper,
):
    p.init_cache(subgrid_size, cell_size, w_step, shift)
    p.degridding(
        kernel_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        aterms,
        aterm_offsets,
        taper,
    )


def main(proxyname, plot=True):
    if plot:
        import matplotlib.pyplot as plt

    """Run example code with any proxy given by 'proxyname'"""

    # Get parameters from global variables
    nr_channels = NR_CHANNELS
    nr_timesteps = NR_TIMESTEPS
    nr_timeslots = NR_TIMESLOTS
    subgrid_size = SUBGRID_SIZE
    grid_size = GRID_SIZE
    integration_time = INTEGRATION_TIME
    kernel_size = KERNEL_SIZE
    nr_correlations = NR_CORRELATIONS
    w_step = 0.0
    layout_file = LAYOUT_FILE
    nr_stations = NR_STATIONS
    nr_baselines = (nr_stations * (nr_stations - 1)) // 2  # get_nr_baselines()

    # Initialize proxies
    ref = idg.CPU.Reference()
    opt = proxyname()

    # Initialize data
    data = Data(layout_file)

    # Limit baselines in length and number
    max_uv = data.compute_max_uv(grid_size, nr_channels)  # m
    data.limit_max_baseline_length(max_uv)
    data.limit_nr_baselines(nr_baselines)

    # Get remaining parameters
    image_size = data.compute_image_size(grid_size, nr_channels)
    cell_size = image_size / grid_size

    # Print parameters
    print("nr_stations           = ", nr_stations)
    print("nr_baselines          = ", nr_baselines)
    print("nr_channels           = ", nr_channels)
    print("nr_timesteps          = ", nr_timesteps)
    print("nr_timeslots          = ", nr_timeslots)
    print("nr_correlations       = ", nr_correlations)
    print("subgrid_size          = ", subgrid_size)
    print("grid_size             = ", grid_size)
    print("image_size            = ", image_size)
    print("kernel_size           = ", kernel_size)

    # Initialize data
    channel_offset = 0
    baseline_offset = 0
    time_offset = 0

    uvw = np.zeros((nr_baselines, nr_timesteps, 3), dtype=np.float32)
    frequencies = np.zeros(nr_channels, dtype=idg.frequenciestype)
    data.get_frequencies(frequencies, nr_channels, image_size, channel_offset)
    data.get_uvw(
        uvw, nr_baselines, nr_timesteps, baseline_offset, time_offset, integration_time
    )

    baselines = util.get_example_baselines(nr_stations, nr_baselines)
    aterms = util.get_example_aterms(
        nr_timeslots, nr_stations, subgrid_size, nr_correlations
    )
    aterm_offsets = util.get_example_aterm_offsets(nr_timeslots, nr_timesteps)
    taper = util.get_identity_taper(subgrid_size)
    shift = np.zeros(2, dtype=np.float32)

    # set w to zero
    uvw[...,-1] = 0

    # Initialize dummy visibilities
    example_visibilities = util.get_example_visibilities(
        nr_baselines,
        nr_timesteps,
        nr_channels,
        nr_correlations,
        image_size,
        grid_size,
        uvw,
        frequencies,
    )

    # Initialize empty grids and visibilities
    ref_grid = util.get_zero_grid(nr_correlations, grid_size)
    opt_grid = util.get_zero_grid(nr_correlations, grid_size)
    ref_visibilities = util.get_zero_visibilities(
        nr_baselines, nr_timesteps, nr_channels, nr_correlations
    )
    opt_visibilities = util.get_zero_visibilities(
        nr_baselines, nr_timesteps, nr_channels, nr_correlations
    )

    ref.set_grid(ref_grid)
    opt.set_grid(opt_grid)

    # Run gridding
    for gridder in [ref, opt]:
        gridding(
            gridder,
            w_step,
            shift,
            cell_size,
            kernel_size,
            subgrid_size,
            frequencies,
            example_visibilities,
            uvw,
            baselines,
            aterms,
            aterm_offsets,
            taper,
        )

    diff_grid = opt_grid - ref_grid
    nnz_grid = len(abs(diff_grid) > 0)
    diff_grid = diff_grid / max(1, nnz_grid)

    if plot:
        # Plot difference beween grids
        util.plot_grid(opt_grid, scaling="log")
        util.plot_grid(ref_grid, scaling="log")
        util.plot_grid(diff_grid)

    # Run degridding
    for degridder, visibilities in zip(
        [ref, opt], [ref_visibilities, opt_visibilities]
    ):
        degridding(
            degridder,
            w_step,
            shift,
            cell_size,
            kernel_size,
            subgrid_size,
            frequencies,
            visibilities,
            uvw,
            baselines,
            aterms,
            aterm_offsets,
            taper,
        )

    diff_visibilities = opt_visibilities - ref_visibilities
    nnz_visibilities = len(abs(diff_visibilities) > 0)
    diff_visibilities = diff_visibilities / max(1, nnz_visibilities)

    if plot:
        # Plot difference between visibilities
        util.plot_visibilities(ref_visibilities)
        util.plot_visibilities(opt_visibilities)
        util.plot_visibilities(diff_visibilities)
        plt.show()

    return ref_grid, ref_visibilities, opt_grid, opt_visibilities
