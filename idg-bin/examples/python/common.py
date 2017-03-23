# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
import idg
import utils
import numpy
import matplotlib.pyplot as plt
import random

############
# paramaters
############
_nr_stations      = 50
_nr_baselines     = _nr_stations*(_nr_stations-1)/2
_nr_channels      = 16
_nr_timesteps     = 1024           # samples per baseline
_nr_timeslots     = 16             # A-term time slots
_image_size       = 0.05
_subgrid_size     = 24
_grid_size        = 2048
_integration_time = 1
_kernel_size      = (_subgrid_size / 2) + 1
_nr_correlations  = 4

def get_nr_stations():
    return _nr_stations

def get_nr_baselines():
    return _nr_baselines

def get_nr_channels():
    return _nr_channels

def get_nr_timesteps():
    return _nr_timesteps

def get_nr_timeslots():
    return _nr_timeslots

def get_image_size():
    return _image_size

def get_subgrid_size():
    return _subgrid_size

def get_cell_size():
    return _image_size / _grid_size

def get_grid_size():
    return _grid_size

def get_integration_time():
    return _integration_time

def get_kernel_size():
    return _kernel_size

def get_nr_correlations():
    return _nr_correlations

#################
# initialize data
#################
def init_dummy_visibilities(nr_baselines, nr_timesteps, nr_channels):
    visibilities =  numpy.ones(
        (nr_baselines, nr_timesteps, nr_channels, _nr_correlations),
        dtype = idg.visibilitiestype)
    #utils.plot_visibilities(visibilities)
    return visibilities


###########
# debugging
###########
def plot_metadata(
        kernel_size, subgrid_size, grid_size, cell_size, image_size,
        frequencies, uvw, baselines, aterms_offsets, wavenumbers,
        max_nr_timesteps = numpy.iinfo(numpy.int32).max):
    plan = idg.Plan(
        kernel_size, subgrid_size, grid_size, cell_size,
        frequencies, uvw, baselines, aterms_offsets, max_nr_timesteps)
    nr_subgrids = plan.get_nr_subgrids()
    metadata = numpy.zeros(nr_subgrids, dtype = idg.metadatatype)
    plan.copy_metadata(metadata)
    utils.plot_metadata(metadata, uvw, wavenumbers, grid_size, subgrid_size, image_size)


##########
# gridding
##########
def gridding(
        p, w_offset, cell_size, kernel_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterms_offsets, spheroidal):
    p.gridding(
        w_offset, cell_size, kernel_size,
        frequencies, visibilities, uvw, baselines,
        grid, aterms, aterms_offsets, spheroidal)
    utils.plot_grid(grid, scaling='log')
    p.transform(idg.FourierDomainToImageDomain, grid)
    utils.plot_grid(grid)
    #utils.plot_grid(grid, pol=0)


############
# degridding
############
def degridding(
        p, w_offset, cell_size, kernel_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterms_offsets, spheroidal):
    p.transform(idg.ImageDomainToFourierDomain, grid)
    p.degridding(
        w_offset, cell_size, kernel_size,
        frequencies, visibilities, uvw, baselines,
        grid, aterms, aterms_offsets, spheroidal)
    #utils.plot_visibilities(visibilities)


def main(proxyname):
    """Run example code with any proxy given by 'proxyname'"""

    ######################################################################
    # Set parameters
    ######################################################################
    nr_stations      = get_nr_stations()
    nr_baselines     = get_nr_baselines()
    nr_channels      = get_nr_channels()
    nr_timesteps     = get_nr_timesteps()
    nr_timeslots     = get_nr_timeslots()
    image_size       = get_image_size()
    cell_size        = get_cell_size()
    subgrid_size     = get_subgrid_size()
    grid_size        = get_grid_size()
    integration_time = get_integration_time()
    kernel_size      = get_kernel_size()
    nr_correlations  = get_nr_correlations()
    w_offset         = 0.0

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
    uvw            = utils.get_example_uvw(
                        nr_baselines, nr_timesteps, integration_time)
    wavenumbers    = utils.get_example_wavenumbers(nr_channels)
    frequencies    = utils.get_example_frequencies(nr_channels)
    baselines      = utils.get_example_baselines(nr_baselines)
    grid           = utils.get_example_grid(
                        nr_correlations, grid_size)
    aterms         = utils.get_example_aterms(
                        nr_timeslots, nr_stations, subgrid_size, nr_correlations)
    aterms_offsets = utils.get_example_aterms_offset(
                        nr_timeslots, nr_timesteps)
    spheroidal     = utils.get_identity_spheroidal(subgrid_size)
    visibilities   = utils.get_example_visibilities(
                        nr_baselines, nr_timesteps, nr_channels, nr_correlations,
                        image_size, grid_size, uvw, wavenumbers)

    ######################################################################
    # plot data
    ######################################################################
    # utils.plot_uvw(uvw)
    # utils.plot_wavenumbers(wavenumbers)
    # utils.plot_spheroidal(spheroidal)
    # utils.plot_visibilities(visibilities)
    # plot_metadata(
    #     kernel_size, subgrid_size, grid_size, cell_size, image_size,
    #     frequencies, uvw, baselines, aterms_offsets, wavenumbers)

    ######################################################################
    # routines
    ######################################################################
    gridding(
        p, w_offset, cell_size, kernel_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterms_offsets, spheroidal)

    degridding(
        p, w_offset, cell_size, kernel_size, frequencies, visibilities,
        uvw, baselines, grid, aterms, aterms_offsets, spheroidal)

    plt.show()
