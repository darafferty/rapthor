#!/usr/bin/env python

# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
import idg
import numpy
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ############
    # paramaters
    ############
    nr_stations = 8
    nr_baselines = nr_stations*(nr_stations-1)/2
    nr_channels = 1
    nr_time = 4800            # samples per baseline
    nr_timeslots = 10         # A-term time slots
    image_size = 0.08
    subgrid_size = 24
    grid_size = 1024
    integration_time = 10
    kernel_size = (subgrid_size / 2) + 1
    w_offset = 0

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
    #idg.utils.plot_uvw(uvw)

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
    spheroidal = numpy.ones((subgrid_size, subgrid_size),
                             dtype = idg.spheroidaltype)
    idg.utils.init_spheroidal(spheroidal)
    #idg.utils.plot_spheroidal(spheroidal)


    #
    # Add point source in the middle of the grid
    #
    grid[:,grid_size/2,grid_size/2] = 1

    
    ############
    # degridding
    ############
    grid = numpy.fft.ifftshift(grid, axes=(1,2))
    p.transform(idg.ImageDomainToFourierDomain, grid)
    grid = numpy.fft.fftshift(grid, axes=(1,2))
    p.degrid_visibilities(visibilities, uvw, wavenumbers, baselines, grid,
                          w_offset, kernel_size, aterms, aterms_offset, spheroidal)
    idg.utils.plot_visibilities(visibilities)

    # reset grid
    grid = numpy.zeros((nr_polarizations, grid_size, grid_size),
                       dtype = idg.gridtype)

    ##########
    # gridding
    ##########
    p.grid_visibilities(visibilities, uvw, wavenumbers, baselines, grid,
                        w_offset, kernel_size, aterms, aterms_offset, spheroidal)
    grid = numpy.fft.ifftshift(grid, axes=(1,2))
    p.transform(idg.FourierDomainToImageDomain, grid)
    grid = numpy.fft.fftshift(grid, axes=(1,2))

    # plot result (should look like the point spread function)
    idg.utils.plot_grid(grid)

    plt.show()
