#!/usr/bin/env python

# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
import idg
import numpy
import matplotlib.pyplot as plt
import sys
import signal


def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    ############
    # paramaters
    ############
    nr_stations = 15
    nr_baselines = nr_stations*(nr_stations-1)/2
    nr_channels = 8
    nr_timesteps = 16
    nr_timeslots = 300
    nr_time = nr_timesteps*nr_timeslots
    image_size = 0.008
    subgrid_size = 24
    grid_size = 1024
    integration_time = 10

    ##################
    # initialize proxy
    ##################
    p = idg.CPU.HaswellEP(nr_stations, nr_channels,
                          nr_timesteps, nr_timeslots,
                          image_size, grid_size, subgrid_size)
    ##################
    # print parameters
    ##################
    print "Proxy: nr_stations = ", p.get_nr_stations()
    print "Proxy: nr_baselines = ", p.get_nr_baselines()
    print "Proxy: nr_channels = ", p.get_nr_channels()
    print "Proxy: nr_timesteps = ", p.get_nr_timesteps()
    print "Proxy: nr_timeslots = ", p.get_nr_timeslots()
    print "Proxy: nr_polarizations = ", p.get_nr_polarizations()
    print "Proxy: subgrid_size = ", p.get_subgrid_size()
    print "Proxy: grid_size = ", p.get_grid_size()
    print "Proxy: image_size = ", p.get_image_size()
    print "Proxy: job size for gridding = ", p.get_job_size_gridding()
    print "Proxy: job size for degridding = ", p.get_job_size_degridding()

    #################
    # initialize data
    #################

    # allocate memory for data
    nr_polarizations = p.get_nr_polarizations()

    # visibilities
    visibilities =  numpy.ones(
        (nr_baselines, nr_time, nr_channels, nr_polarizations),
        dtype = idg.visibilitiestype)
    idg.utils.init_visibilities(visibilities)
    idg.utils.plot_visibilities(visibilities)

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

    # aterm
    aterms = numpy.zeros((nr_stations, nr_timeslots, nr_polarizations,
                          subgrid_size, subgrid_size), \
                         dtype = idg.atermtype)
    # idg.utils.init_aterms(aterms)
    # TODO: update C++ init_aterms
    # Set aterm to identity instead
    aterms[:,:,0,:,:] = 1.0
    aterms[:,:,3,:,:] = 1.0

    # spheroidal
    spheroidal = numpy.ones((subgrid_size, subgrid_size),
                             dtype = idg.spheroidaltype)
    idg.utils.init_spheroidal(spheroidal)
    #idg.utils.plot_spheroidal(spheroidal)

    # metadata (for debugging)
    nr_subgrids = p._get_nr_subgrids(uvw, wavenumbers, baselines)
    metadata = numpy.zeros(nr_subgrids, dtype = idg.metadatatype)
    p._init_metadata(metadata, uvw, wavenumbers, baselines)
    idg.utils.plot_metadata(metadata, uvw, wavenumbers, grid_size, subgrid_size, image_size)

    ##########
    # gridding
    ##########
    w_offset = 0.0

    p.grid_visibilities(visibilities, uvw, wavenumbers, baselines, grid,
                        w_offset, aterms, spheroidal)
    idg.utils.plot_grid(grid, scaling='log')

    # TODO: shift zero frequency to outer part
    grid = numpy.fft.ifftshift(grid, axes=(1,2))
    #idg.utils.plot_grid(grid)

    p.transform(idg.FourierDomainToImageDomain, grid)

    # TODO: move to fft routine
    grid.real *= 2
    grid.imag = 0

    grid = numpy.fft.fftshift(grid, axes=(1,2))
    idg.utils.plot_grid(grid)

    ############
    # degridding
    ############
    grid = numpy.fft.ifftshift(grid, axes=(1,2))
    p.transform(idg.ImageDomainToFourierDomain, grid)

    # TODO: Shift the zero-frequency component to the center of the spectrum.
    grid = numpy.fft.fftshift(grid, axes=(1,2))
    #idg.utils.plot_grid(grid)

    p.degrid_visibilities(visibilities, uvw, wavenumbers, baselines, grid,
                          w_offset, aterms, spheroidal)
    idg.utils.plot_visibilities(visibilities)

    plt.show()
