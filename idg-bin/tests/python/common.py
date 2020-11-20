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
_nr_timesteps     = 1*60*60        # samples per baseline
_nr_timeslots     = 16             # A-term time slots
_subgrid_size     = 32
_grid_size        = 2048
_integration_time = 0.9
_kernel_size      = 9
_nr_correlations  = 4
_layout_file      = "SKA1_low_ecef"
_nr_stations      = 12
_nr_baselines     = (_nr_stations * (_nr_stations - 1)) / 2

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
        frequencies, uvw, baselines, aterms_offsets,
        max_nr_timesteps = numpy.iinfo(numpy.int32).max):
    plan = idg.Plan(
        kernel_size, subgrid_size, grid_size, cell_size,
        frequencies, uvw, baselines, aterms_offsets, max_nr_timesteps)
    nr_subgrids = plan.get_nr_subgrids()
    metadata = numpy.zeros(nr_subgrids, dtype = idg.metadatatype)
    plan.copy_metadata(metadata)
    util.plot_metadata(metadata, uvw, frequencies, grid_size, subgrid_size, image_size)



##########
# gridding
##########
def gridding(
        p, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterms_offsets, spheroidal):
    p.gridding(
        w_step, shift, cell_size, kernel_size, subgrid_size,
        frequencies, visibilities, uvw, baselines,
        grid, aterms, aterms_offsets, spheroidal)
    #p.transform(idg.FourierDomainToImageDomain, grid)


############
# degridding
############
def degridding(
        p, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterms_offsets, spheroidal):
    p.transform(idg.ImageDomainToFourierDomain, grid)
    p.degridding(
        w_step, shift, cell_size, kernel_size, subgrid_size,
        frequencies, visibilities, uvw, baselines,
        grid, aterms, aterms_offsets, spheroidal)


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

    ######################################################################
    # initialize proxies
    ######################################################################
    ref = idg.CPU.Reference(nr_correlations, subgrid_size)
    p = ref
    opt = proxyname(nr_correlations, subgrid_size)

    ######################################################################
    # initialize data
    ######################################################################
    data                  = Data(layout_file)

    # Limit baselines in length and number
    max_uv = data.compute_max_uv(int(grid_size)) # m
    data.limit_max_baseline_length(max_uv)
    data.limit_nr_baselines(nr_baselines)

    # Get remaining parameters
    image_size = data.compute_image_size(grid_size)
    cell_size  = image_size / grid_size

    ######################################################################
    # print parameters
    ######################################################################
    print "nr_stations           = ", nr_stations
    print "nr_baselines          = ", nr_baselines
    print "nr_channels           = ", nr_channels
    print "nr_timesteps          = ", nr_timesteps
    print "nr_timeslots          = ", nr_timeslots
    print "nr_correlations       = ", nr_correlations
    print "subgrid_size          = ", subgrid_size
    print "grid_size             = ", grid_size
    print "image_size            = ", image_size
    print "kernel_size           = ", kernel_size

    ######################################################################
    # initialize data
    ######################################################################
    channel_offset  = 0
    baseline_offset = 0
    time_offset     = 0

    uvw            = numpy.zeros((nr_baselines, nr_timesteps), dtype=idg.uvwtype)
    frequencies    = numpy.zeros((nr_channels), dtype=idg.frequenciestype)
    data.get_frequencies(frequencies, nr_channels, image_size, channel_offset)
    data.get_uvw(uvw, nr_baselines, nr_timesteps, baseline_offset, time_offset, integration_time)

    baselines      = util.get_example_baselines(nr_baselines)
    aterms         = util.get_example_aterms(
                        nr_timeslots, nr_stations, subgrid_size, nr_correlations)
    aterms_offsets = util.get_example_aterms_offset(
                        nr_timeslots, nr_timesteps)
    spheroidal     = util.get_identity_spheroidal(subgrid_size)
    shift          = numpy.zeros(3, dtype=float)

    uvw['w'] = 0
    ######################################################################
    # initialize visibilities
    ######################################################################
    example_visibilities = util.get_example_visibilities(
                        nr_baselines, nr_timesteps, nr_channels, nr_correlations,
                        image_size, grid_size, uvw, frequencies)

    ######################################################################
    # initialize empty grids and visibilities
    ######################################################################
    ref_grid = util.get_zero_grid(nr_correlations, grid_size)
    opt_grid = util.get_zero_grid(nr_correlations, grid_size)
    ref_visibilities = util.get_zero_visibilities(nr_baselines, nr_timesteps, nr_channels, nr_correlations)
    opt_visibilities = util.get_zero_visibilities(nr_baselines, nr_timesteps, nr_channels, nr_correlations)

    ######################################################################
    # run gridding
    ######################################################################
    gridding(
        ref, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, example_visibilities,
        uvw, baselines, opt_grid, aterms, aterms_offsets, spheroidal)

    gridding(
        opt, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, example_visibilities,
        uvw, baselines, ref_grid, aterms, aterms_offsets, spheroidal)

    ######################################################################
    # plot difference between grids
    ######################################################################
    util.plot_grid(opt_grid, scaling='log')
    util.plot_grid(ref_grid, scaling='log')
    diff_grid = opt_grid - ref_grid
    nnz = len(abs(diff_grid) > 0)
    diff_grid = diff_grid / max(1, nnz)
    util.plot_grid(diff_grid)
    #plt.show()

    ######################################################################
    # run degridding
    ######################################################################
    degridding(
        opt, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, ref_visibilities,
        uvw, baselines, ref_grid, aterms, aterms_offsets, spheroidal)

    degridding(
        ref, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, opt_visibilities,
        uvw, baselines, opt_grid, aterms, aterms_offsets, spheroidal)

    ######################################################################
    # plot difference between visibilities
    ######################################################################
    util.plot_visibilities(ref_visibilities)
    util.plot_visibilities(opt_visibilities)
    diff_visibilities = opt_visibilities - ref_visibilities
    nnz = len(abs(diff_visibilities) > 0)
    diff_visibilities = diff_visibilities / max(1, nnz)
    util.plot_visibilities(diff_visibilities)

    plt.show()
