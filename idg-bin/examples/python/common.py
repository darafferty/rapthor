#!/usr/bin/env python
# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
import idg
import idg.util as util
from idg.data import Data
import numpy
import matplotlib.pyplot as plt
import random

############
# paramaters
############
_nr_channels      = 1
_nr_timesteps     = 2*60*60        # samples per baseline
_nr_timeslots     = 16             # A-term time slots
_subgrid_size     = 24
_grid_size        = 2048
_integration_time = 0.9
_kernel_size      = int((_subgrid_size / 2) + 1)
_nr_correlations  = 4
_layout_file      = "LOFAR_lba.txt"
_nr_stations      = 20
_nr_baselines     = int((_nr_stations * (_nr_stations - 1)) / 2)

def get_nr_channels():
    return _nr_channels

def get_nr_timesteps():
    return _nr_timesteps

def get_nr_timeslots():
    return _nr_timeslots

def get_subgrid_size():
    return _subgrid_size

def get_grid_size():
    return _grid_size

def get_integration_time():
    return _integration_time

def get_kernel_size():
    return _kernel_size

def get_nr_correlations():
    return _nr_correlations

def get_layout_file():
    return _layout_file

def get_nr_stations():
    return _nr_stations

def get_nr_baselines():
    return _nr_baselines


###########
# debugging
###########
def plot_metadata(
        kernel_size, subgrid_size, grid_size, cell_size, image_size,
        frequencies, uvw, baselines, aterm_offsets):
    plan = idg.Plan(
        kernel_size, subgrid_size, grid_size, cell_size,
        frequencies, uvw, baselines, aterm_offsets)
    nr_subgrids = plan.get_nr_subgrids()
    metadata = numpy.zeros(nr_subgrids, dtype = idg.metadatatype)
    plan.copy_metadata(metadata)
    util.plot_metadata(metadata, uvw, frequencies, grid_size, subgrid_size, image_size)


##########
# gridding
##########
def gridding(
        p, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterm_offsets, taper):
    p.gridding(
        kernel_size, frequencies, visibilities, uvw, baselines,
        aterms, aterm_offsets, taper)
    p.get_final_grid(grid)
    util.plot_grid(grid, scaling='log')
    p.transform(idg.FourierDomainToImageDomain)
    p.get_final_grid(grid)
    util.plot_grid(grid)
    #util.plot_grid(grid, pol=0)


############
# degridding
############
def degridding(
        p, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterm_offsets, taper):
    p.transform(idg.ImageDomainToFourierDomain)
    p.degridding(
        kernel_size, frequencies, visibilities, uvw, baselines,
        aterms, aterm_offsets, taper)
    #util.plot_visibilities(visibilities)


def main(proxyname):
    """Run example code with any proxy given by 'proxyname'"""

    ######################################################################
    # Set parameters
    ######################################################################
    nr_channels      = get_nr_channels()
    nr_timesteps     = get_nr_timesteps()
    nr_timeslots     = get_nr_timeslots()
    subgrid_size     = get_subgrid_size()
    grid_size        = get_grid_size()
    integration_time = get_integration_time()
    kernel_size      = get_kernel_size()
    nr_correlations  = get_nr_correlations()
    w_step           = 0.0
    layout_file      = get_layout_file()
    nr_stations      = get_nr_stations()
    nr_baselines     = get_nr_baselines()
    nr_polarizations = 4 if nr_correlations == 4 else 1

    ######################################################################
    # initialize data generator
    ######################################################################

    # Initialize full dataset
    data = Data(layout_file)
    print(">> Dataset full: ")
    data.print_info()

    # Determine the maximum suggested grid_size using this dataset
    grid_size_max = data.compute_grid_size()
    print("maximum grid size: %d" % (grid_size_max))

    # Determine the max baseline length for given grid_size
    max_uv = data.compute_max_uv(grid_size, nr_channels) # m
    print("longest baseline required: %.2f km" % (max_uv * 1e-3))

    # Select only baselines up to max_uv meters long
    data.limit_max_baseline_length(max_uv)
    print(">> Dataset limited to baseline up to %.2f km: " % (max_uv * 1e-3))
    data.print_info()

    # Restrict the number of baselines to nr_baselines
    data.limit_nr_baselines(nr_baselines)
    print(">> Dataset limited to %d baselines: " % (nr_baselines))
    data.print_info()

    # Get remaining parameters
    image_size = round(data.compute_image_size(grid_size, nr_channels), 4)
    cell_size  = image_size / grid_size


    ######################################################################
    # initialize proxy
    ######################################################################
    p = proxyname()

    ######################################################################
    # print parameters
    ######################################################################
    print("nr_stations           = ", nr_stations)
    print("nr_baselines          = ", nr_baselines)
    print("nr_channels           = ", nr_channels)
    print("nr_timesteps          = ", nr_timesteps)
    print("nr_timeslots          = ", nr_timeslots)
    print("nr_correlations       = ", nr_correlations)
    print("nr_polarizations      = ", nr_polarizations)
    print("subgrid_size          = ", subgrid_size)
    print("grid_size             = ", grid_size)
    print("image_size            = ", image_size)
    print("kernel_size           = ", kernel_size)
    print("integration_time      = ", integration_time)

    ######################################################################
    # initialize data
    ######################################################################
    channel_offset  = 0
    baseline_offset = 0
    time_offset     = 0

    uvw            = numpy.zeros((nr_baselines, nr_timesteps, 3), dtype=numpy.float32)
    frequencies    = numpy.zeros((nr_channels), dtype=numpy.float32)
    data.get_frequencies(frequencies, nr_channels, image_size, channel_offset)
    data.get_uvw(uvw, nr_baselines, nr_timesteps, baseline_offset, time_offset, integration_time)

    baselines      = util.get_example_baselines(nr_stations, nr_baselines)
    grid           = p.allocate_grid(nr_polarizations, grid_size)

    aterms         = util.get_identity_aterms(
                        nr_timeslots, nr_stations, subgrid_size, 4)
    aterm_offsets = util.get_example_aterm_offsets(
                        nr_timeslots, nr_timesteps)
    taper     = util.get_identity_taper(subgrid_size)
    visibilities   = util.get_example_visibilities(
                        nr_baselines, nr_timesteps, nr_channels, nr_correlations,
                        image_size, grid_size, uvw, frequencies)
    shift          = numpy.zeros(2, dtype=numpy.float32)

    ######################################################################
    # plot data
    ######################################################################
    # util.plot_uvw_meters(uvw)
    # util.plot_uvw_pixels(uvw, frequencies, image_size)
    # util.plot_tiles(uvw, frequencies, image_size, grid_size, 128)
    # util.plot_frequencies(frequencies)
    # util.plot_taper(taper)
    # util.plot_visibilities(visibilities)
    # plot_metadata(
    #     kernel_size, subgrid_size, grid_size, cell_size, image_size,
    #     frequencies, uvw, baselines, aterm_offsets)

    ######################################################################
    # routines
    ######################################################################
    p.init_cache(subgrid_size, cell_size, w_step, shift)

    gridding(
        p, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterm_offsets, taper)

    degridding(
        p, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterm_offsets, taper)

    plt.show()
