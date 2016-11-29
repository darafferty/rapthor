# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
import idg
import numpy
import matplotlib.pyplot as plt
import random

############
# paramaters
############
_nr_stations      = 100
_nr_baselines     = _nr_stations*(_nr_stations-1)/2
_nr_channels      = 16
_nr_time          = 1024           # samples per baseline
_nr_timeslots     = 16             # A-term time slots
_image_size       = 0.05
_subgrid_size     = 24
_grid_size        = 2048
_integration_time = 1
_kernel_size      = (_subgrid_size / 2) + 1
_nr_polarizations = 4

def get_nr_stations():
    return _nr_stations

def get_nr_baselines():
    return _nr_baselines

def get_nr_channels():
    return _nr_channels

def get_nr_time():
    return _nr_time

def get_nr_timeslots():
    return _nr_timeslots

def get_image_size():
    return _image_size

def get_subgrid_size():
    return _subgrid_size

def get_grid_size():
    return _grid_size

def get_integration_time():
    return _integration_time

def get_kernel_size():
    return _kernel_size

def get_nr_polarizations():
    return _nr_polarizations

#################
# initialize data
#################
def init_visibilities_dummy(nr_baselines, nr_time, nr_channels):
    visibilities =  numpy.ones(
        (nr_baselines, nr_time, nr_channels, _nr_polarizations),
        dtype = idg.visibilitiestype)
    idg.utils.init_visibilities(visibilities)
    #idg.utils.plot_visibilities(visibilities)
    return visibilities


###########
# debugging
###########
def plot_metadata(
        p, uvw, wavenumbers, baselines, aterms_offset,
        kernel_size, grid_size, subgrid_size, image_size):
    nr_subgrids = p._get_nr_subgrids(uvw, wavenumbers, baselines, aterms_offset, kernel_size)
    metadata = numpy.zeros(nr_subgrids, dtype = idg.metadatatype)
    p._init_metadata(metadata, uvw, wavenumbers, baselines, aterms_offset, kernel_size)
    idg.utils.plot_metadata(metadata, uvw, wavenumbers, grid_size, subgrid_size, image_size)


##########
# gridding
##########
def gridding(
        p, visibilities, uvw, wavenumbers, baselines, grid,
        kernel_size, aterms, aterms_offset, spheroidal):
    w_offset = 0.0
    p.grid_visibilities(
        visibilities, uvw, wavenumbers, baselines, grid,
        w_offset, kernel_size, aterms, aterms_offset, spheroidal)
    idg.utils.plot_grid(grid, scaling='log')
    p.transform(idg.FourierDomainToImageDomain, grid)
    idg.utils.plot_grid(grid)
    #idg.utils.plot_grid(grid, pol=0)


############
# degridding
############
def degridding(
        p, visibilities, uvw, wavenumbers, baselines, grid,
        kernel_size, aterms, aterms_offset, spheroidal):
    w_offset = 0.0
    p.transform(idg.ImageDomainToFourierDomain, grid)
    p.degrid_visibilities(
        visibilities, uvw, wavenumbers, baselines, grid,
        w_offset, kernel_size, aterms, aterms_offset, spheroidal)
    #idg.utils.plot_visibilities(visibilities)


def main(proxyname):
    """Run example code with any proxy given by 'proxyname'"""

    ######################################################################
    # Set parameters
    ######################################################################
    nr_stations = get_nr_stations()
    nr_baselines = get_nr_baselines()
    nr_channels = get_nr_channels()
    nr_time = get_nr_time()
    nr_timeslots = get_nr_timeslots()
    image_size = get_image_size()
    subgrid_size = get_subgrid_size()
    grid_size = get_grid_size()
    integration_time = get_integration_time()
    kernel_size = get_kernel_size()
    nr_polarizations = get_nr_polarizations()

    ######################################################################
    # initialize proxy
    ######################################################################
    p = proxyname(nr_stations, nr_channels,
                  nr_time, nr_timeslots,
                  image_size, grid_size,
                  subgrid_size)

    ######################################################################
    # print parameters
    ######################################################################
    print "nr_stations           = ", p.get_nr_stations()
    print "nr_baselines          = ", p.get_nr_baselines()
    print "nr_channels           = ", p.get_nr_channels()
    print "nr_timesteps          = ", p.get_nr_time()
    print "nr_timeslots          = ", p.get_nr_timeslots()
    print "nr_polarizations      = ", p.get_nr_polarizations()
    print "subgrid_size          = ", p.get_subgrid_size()
    print "grid_size             = ", p.get_grid_size()
    print "image_size            = ", p.get_image_size()
    print "kernel_size           = ", kernel_size
    print "job size (gridding)   = ", p.get_job_size_gridding()
    print "job size (degridding) = ", p.get_job_size_degridding()

    ######################################################################
    # initialize data
    ######################################################################
    uvw           = idg.utils.get_example_uvw(nr_baselines, nr_time,
                                              integration_time)
    wavenumbers   = idg.utils.get_example_wavenumbers(nr_channels)
    baselines     = idg.utils.get_example_baselines(nr_baselines)
    grid          = idg.utils.get_example_grid(nr_polarizations,
                                               grid_size)
    aterms        = idg.utils.get_example_aterms(nr_timeslots, nr_stations,
                                                 subgrid_size,
                                                 nr_polarizations)
    aterms_offset = idg.utils.get_example_aterms_offset(nr_timeslots,
                                                        nr_time)
    spheroidal    = idg.utils.get_identity_spheroidal(subgrid_size)
    visibilities  = idg.utils.get_example_visibilities(nr_baselines,
                                                       nr_time,
                                                       nr_channels,
                                                       nr_polarizations,
                                                       image_size,
                                                       grid_size,
                                                       uvw,
                                                       wavenumbers)

    ######################################################################
    # plot data
    ######################################################################
    # idg.utils.plot_uvw(uvw)
    # idg.utils.plot_wavenumbers(wavenumbers)
    # idg.utils.plot_spheroidal(spheroidal)
    # idg.utils.plot_visibilities(visibilities)
    # plot_metadata(
    #    p, uvw, wavenumbers, baselines, aterms_offset,
    #    kernel_size, grid_size, subgrid_size, image_size)

    ######################################################################
    # routines
    ######################################################################
    gridding(
        p, visibilities, uvw, wavenumbers, baselines, grid,
        kernel_size, aterms, aterms_offset, spheroidal)

    degridding(
        p, visibilities, uvw, wavenumbers, baselines, grid,
        kernel_size, aterms, aterms_offset, spheroidal)

    plt.show()
