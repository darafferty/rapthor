#!/usr/bin/env python

# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
import idg
import numpy
import matplotlib.pyplot as plt
import sys
import signal
import scipy.misc


def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    #################
    # initialize lena
    #################
    lena = scipy.misc.lena()
    lena_img = numpy.matrix(lena, dtype = numpy.complex64)
    lena_freq = numpy.fft.fftshift(numpy.fft.fft2(lena_img))
    plt.gray()

    ############
    # paramaters
    ############
    nr_stations = 30
    nr_baselines = nr_stations*(nr_stations-1)/2
    nr_channels = 1
    nr_time = 4800            # samples per baseline
    nr_timeslots = 10         # A-term time slots
    image_size = 0.08
    image_size = 0.10
    subgrid_size = 24
    grid_size = lena_img.shape[0]
    integration_time = 10
    kernel_size = (subgrid_size / 2) + 1

    ##################
    # initialize proxy
    ##################
    p = idg.CPU.Reference(nr_stations, nr_channels,
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

    # allocate memory for data
    nr_polarizations = p.get_nr_polarizations()

    # visibilities
    visibilities =  numpy.zeros(
        (nr_baselines, nr_time, nr_channels, nr_polarizations),
        dtype = idg.visibilitiestype)

    # uvw
    uvw = numpy.zeros((nr_baselines, nr_time),
                      dtype = idg.uvwtype)
    idg.utils.init_uvw(uvw, integration_time)
    idg.utils.plot_uvw(uvw)

    # wavenumbers
    wavenumbers = numpy.ones(nr_channels,
                             dtype = idg.wavenumberstype)
    idg.utils.init_wavenumbers(wavenumbers)
    #idg.utils.plot_wavenumbers(wavenumbers)

    # baselines
    baselines = numpy.zeros(nr_baselines, dtype = idg.baselinetype)
    idg.utils.init_baselines(baselines)

    # grid
    grid = numpy.zeros((nr_polarizations, grid_size, grid_size),
                       dtype = idg.gridtype)
    grid[:,:,:] = lena_img
    idg.utils.plot_grid(grid, scaling='log')
    #grid[:,:,:] = lena_freq
    p.transform(idg.ImageDomainToFourierDomain, grid)
    idg.utils.plot_grid(grid, scaling='log')

    # aterms
    aterms = numpy.zeros((nr_stations, nr_timeslots, nr_polarizations,
                          subgrid_size, subgrid_size), \
                         dtype = idg.atermtype)
    # idg.utils.init_aterms(aterms)
    # TODO: update C++ init_aterms
    # Set aterm to identity instead
    aterms[:,:,0,:,:] = 1.0
    aterms[:,:,3,:,:] = 1.0

    # aterm offset
    aterms_offset = numpy.zeros((nr_timeslots + 1), dtype = idg.atermoffsettype)
    idg.utils.init_aterms_offset(aterms_offset, nr_time)

    # spheroidal
    #spheroidal = numpy.ones((subgrid_size, subgrid_size),
    #                         dtype = idg.spheroidaltype)
    #idg.utils.init_spheroidal(spheroidal)
    spheroidal_subgrid = idg.utils.init_spheroidal_subgrid(subgrid_size)
    spheroidal_grid = idg.utils.init_spheroidal_grid(subgrid_size, grid_size)
    idg.utils.plot_spheroidal(spheroidal_subgrid)
    idg.utils.plot_spheroidal(spheroidal_grid)

    # metadata (for debugging)
    #nr_subgrids = p._get_nr_subgrids(uvw, wavenumbers, baselines, aterms_offset, kernel_size)
    #metadata = numpy.zeros(nr_subgrids, dtype = idg.metadatatype)
    #p._init_metadata(metadata, uvw, wavenumbers, baselines, aterms_offset, kernel_size)
    #idg.utils.plot_metadata(metadata, uvw, wavenumbers, grid_size, subgrid_size, image_size)

    ############
    # degridding
    ############
    w_offset = 0.0
    #p.transform(idg.ImageDomainToFourierDomain, grid)

    p.degrid_visibilities(visibilities, uvw, wavenumbers, baselines, grid,
                          w_offset, kernel_size, aterms, aterms_offset, spheroidal_subgrid)
    #idg.utils.plot_visibilities(visibilities)

    ##########
    # gridding
    ##########
    grid = numpy.zeros((nr_polarizations, grid_size, grid_size),
                       dtype = idg.gridtype)
    p.grid_visibilities(visibilities, uvw, wavenumbers, baselines, grid,
                        w_offset, kernel_size, aterms, aterms_offset, spheroidal_subgrid)

    idg.utils.plot_grid(grid, scaling='log')

    p.transform(idg.FourierDomainToImageDomain, grid)

    idg.utils.plot_grid(grid)

    plt.show()
