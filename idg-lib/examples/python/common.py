# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
import idg
import numpy
import matplotlib.pyplot as plt

############
# paramaters
############
_nr_stations = 8
_nr_baselines = _nr_stations*(_nr_stations-1)/2
_nr_channels = 1
_nr_time = 4800            # samples per baseline
_nr_timeslots = 10         # A-term time slots
_image_size = 0.08
_subgrid_size = 24
_grid_size = 1024
_integration_time = 10
_kernel_size = (_subgrid_size / 2) + 1
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


#################
# initialize data
#################
def init_visibilities(nr_baselines, nr_time, nr_channels):
    visibilities =  numpy.ones(
        (nr_baselines, nr_time, nr_channels, _nr_polarizations),
        dtype = idg.visibilitiestype)
    idg.utils.init_visibilities(visibilities)
    idg.utils.plot_visibilities(visibilities)
    return visibilities

def init_uvw(nr_baselines, nr_time, integration_time):
    uvw = numpy.zeros((nr_baselines, nr_time),
                      dtype = idg.uvwtype)
    idg.utils.init_uvw(uvw, integration_time)
    idg.utils.plot_uvw(uvw)
    return uvw

def init_wavenumbers(nr_channels):
    wavenumbers = numpy.ones(
        nr_channels,
        dtype = idg.wavenumberstype)
    idg.utils.init_wavenumbers(wavenumbers)
    #idg.utils.plot_wavenumbers(wavenumbers)
    return wavenumbers


def init_baselines(nr_baselines):
    baselines = numpy.zeros(
        nr_baselines,
        dtype = idg.baselinetype)
    idg.utils.init_baselines(baselines)
    return baselines

def init_grid(grid_size):
    grid = numpy.zeros(
        (_nr_polarizations, grid_size, grid_size),
        dtype = idg.gridtype)
    return grid

def init_aterms(nr_stations, nr_timeslots, subgrid_size):
    aterms = numpy.zeros(
        (nr_stations, nr_timeslots, _nr_polarizations, subgrid_size, subgrid_size),
        dtype = idg.atermtype)
    # idg.utils.init_aterms(aterms)

    # TODO: update C++ init_aterms
    # Set aterm to identity instead
    aterms[:,:,0,:,:] = 1.0
    aterms[:,:,3,:,:] = 1.0

    return aterms

def init_aterms_offset(nr_timeslots, nr_time):
    aterms_offset = numpy.zeros(
        (nr_timeslots + 1),
        dtype = idg.atermoffsettype)
    idg.utils.init_aterms_offset(aterms_offset, nr_time)
    return aterms_offset

def init_spheroidal(subgrid_size):
    spheroidal = numpy.ones(
        (subgrid_size, subgrid_size),
        dtype = idg.spheroidaltype)
    idg.utils.init_spheroidal(spheroidal)
    #idg.utils.plot_spheroidal(spheroidal)
    return spheroidal


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
    idg.utils.plot_visibilities(visibilities)
    plt.show()


######
# main
######
def main(proxyname):
    # Set parameters
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

    ##################
    # initialize proxy
    ##################
    p = proxyname(
        nr_stations, nr_channels,
        nr_time, nr_timeslots,
        image_size, grid_size, subgrid_size)

    ##################
    # print parameters
    ##################
    print "nr_stations = ", p.get_nr_stations()
    print "nr_baselines = ", p.get_nr_baselines()
    print "nr_channels = ", p.get_nr_channels()
    print "nr_timeslots = ", p.get_nr_timeslots()
    print "nr_polarizations = ", p.get_nr_polarizations()
    print "subgrid_size = ", p.get_subgrid_size()
    print "grid_size = ", p.get_grid_size()
    print "image_size = ", p.get_image_size()
    print "kernel_size = ", kernel_size
    print "job size for gridding = ", p.get_job_size_gridding()
    print "job size for degridding = ", p.get_job_size_degridding()

    #################
    # initialize data
    #################
    visibilities = init_visibilities(nr_baselines, nr_time, nr_channels)
    uvw = init_uvw(nr_baselines, nr_time, integration_time)
    wavenumbers = init_wavenumbers(nr_channels)
    baselines = init_baselines(nr_baselines)
    grid = init_grid(grid_size)
    aterms = init_aterms(nr_stations, nr_timeslots, subgrid_size)
    aterms_offset = init_aterms_offset(nr_timeslots, nr_time)
    spheroidal = init_spheroidal(subgrid_size)

    ###########
    # debugging
    ###########
    plot_metadata(
        p, uvw, wavenumbers, baselines, aterms_offset,
        kernel_size, grid_size, subgrid_size, image_size)

    ##########
    # routines
    ##########
    gridding(
        p, visibilities, uvw, wavenumbers, baselines, grid,
        kernel_size, aterms, aterms_offset, spheroidal)

    degridding(
        p, visibilities, uvw, wavenumbers, baselines, grid,
        kernel_size, aterms, aterms_offset, spheroidal)

    plt.show()
