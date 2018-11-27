# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
import idg
import util
from data import Data
import numpy
import matplotlib.pyplot as plt
import random

############
# paramaters
############
_nr_channels      = 1
_nr_timesteps     = 1*60*60           # samples per baseline
_nr_timeslots     = 16             # A-term time slots
_subgrid_size     = 24
_integration_time = 0.9
_kernel_size      = (_subgrid_size / 2) + 1
_nr_correlations  = 4
_layout_file           = "SKA1_low_ecef"

def get_nr_channels():
    return _nr_channels

def get_nr_timesteps():
    return _nr_timesteps

def get_nr_timeslots():
    return _nr_timeslots

def get_subgrid_size():
    return _subgrid_size

def get_integration_time():
    return _integration_time

def get_kernel_size():
    return _kernel_size

def get_nr_correlations():
    return _nr_correlations

def get_layout_file():
    return _layout_file

#################
# initialize data
#################
def init_dummy_visibilities(nr_baselines, nr_timesteps, nr_channels):
    visibilities =  numpy.ones(
        (nr_baselines, nr_timesteps, nr_channels, _nr_correlations),
        dtype = idg.visibilitiestype)
    #util.plot_visibilities(visibilities)
    return visibilities


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
    util.plot_grid(grid, scaling='log')
    p.transform(idg.FourierDomainToImageDomain, grid)
    util.plot_grid(grid)
    #util.plot_grid(grid, pol=0)


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
    integration_time = get_integration_time()
    kernel_size      = get_kernel_size()
    nr_correlations  = get_nr_correlations()
    w_step           = 0.0
    layout_file      = get_layout_file()

    ######################################################################
    # initialize data generator
    ######################################################################
    #image_size      = round(data.compute_image_size(int(grid_size*1.2)), 3)

    # Consider all stations and do not restrict baseline length
    nr_stations_limit     = 0
    baseline_length_limit = 0

    # Example 1: use custom grid_size and image_size
    # baseline will be selected to fit within the imposed grid
    # reduce image_size to artificially shrink uv-coverage
    #data                  = Data(nr_stations_limit, baseline_length_limit, layout_file)
    #grid_size             = 1024
    #image_size            = 0.05
    #data.filter_baselines(int(grid_size), image_size)

    # Example 2: specify grid_size
    # this is more realistic, the uv-coverage grows with grid_size
    data                  = Data(nr_stations_limit, baseline_length_limit, layout_file)
    grid_size             = 2048
    padding               = 0.9
    image_size            = round(data.compute_image_size(int(grid_size * padding)), 3)

    # Example 3: specify image_size
    # the grid_size is computed to match the resolution imposed by image_size and baseline length
    #data                  = Data(nr_stations_limit, baseline_length_limit, layout_file)
    #image_size            = 0.05
    #padding               = 1.3
    #grid_size             = int(data.compute_grid_size(image_size) * padding)

    # Example 4: limit max baseline
    # the uv-coverage is controlled by image_size and the baseline_length_limit,
    # using larger grids will not actually increase the resolution of the image
    #baseline_length_limit = 10000
    #data                  = Data(nr_stations_limit, baseline_length_limit, layout_file)
    #grid_size             = 2048
    #image_size            = 0.05

    # Reduce number of stations and baselines to use
    nr_stations           = 20
    nr_baselines          = (nr_stations * (nr_stations - 1)) / 2

    # get remaining parameters
    cell_size             = image_size / grid_size


    ######################################################################
    # initialize proxy
    ######################################################################
    p = proxyname(nr_correlations, subgrid_size)

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
    print "integration_time      = ", integration_time

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
    grid           = p.get_grid(nr_correlations, grid_size)

    aterms         = util.get_identity_aterms(
                        nr_timeslots, nr_stations, subgrid_size, nr_correlations)
    aterms_offsets = util.get_example_aterms_offset(
                        nr_timeslots, nr_timesteps)
    spheroidal     = util.get_identity_spheroidal(subgrid_size)
    visibilities   = util.get_example_visibilities(
                        nr_baselines, nr_timesteps, nr_channels, nr_correlations,
                        image_size, grid_size, uvw, frequencies)
    shift          = numpy.zeros(3, dtype=float)

    ######################################################################
    # plot data
    ######################################################################
    # util.plot_uvw(uvw)
    # util.plot_frequencies(frequencies)
    # util.plot_spheroidal(spheroidal)
    # util.plot_visibilities(visibilities)
    # plot_metadata(
    #     kernel_size, subgrid_size, grid_size, cell_size, image_size,
    #     frequencies, uvw, baselines, aterms_offsets)

    ######################################################################
    # routines
    ######################################################################
    gridding(
        p, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterms_offsets, spheroidal)

    degridding(
        p, w_step, shift, cell_size, kernel_size, subgrid_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterms_offsets, spheroidal)

    plt.show()
