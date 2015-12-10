#!/usr/bin/env python

# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
import idg
import numpy


if __name__ == "__main__":

    nr_stations = 8
    nr_baselines = nr_stations*(nr_stations-1)/2
    nr_channels = 8
    nr_timesteps = 10
    nr_timeslots = 40
    nr_time = nr_timesteps*nr_timeslots
    imagesize = 0.1
    subgrid_size = 8
    grid_size = 128

    p = idg.CPU.Reference(nr_stations, nr_channels,
                          nr_timesteps, nr_timeslots,
                          imagesize, grid_size, subgrid_size)

    print "Proxy: nr_stations = ", p.get_nr_stations()
    print "Proxy: nr_baselines = ", p.get_nr_baselines()
    print "Proxy: nr_channels = ", p.get_nr_channels()
    print "Proxy: nr_timesteps = ", p.get_nr_timesteps()
    print "Proxy: nr_timeslots = ", p.get_nr_timeslots()
    print "Proxy: nr_polarizations = ", p.get_nr_polarizations()
    print "Proxy: nr_subgrid_size = ", p.get_subgrid_size()
    print "Proxy: nr_subgrids = ", p.get_nr_subgrids()
    print "Proxy: nr_grid_size = ", p.get_grid_size()
    print "Proxy: imagesize = ", p.get_imagesize()
    print "Proxy: job size for gridding = ", p.get_job_size_gridding()
    print "Proxy: job size for degridding = ", p.get_job_size_degridding()
    print "Change job size:"
    p.set_job_size_gridding(4096)
    p.set_job_size_degridding(4096)
    print "Proxy: job size for gridding = ", p.get_job_size_gridding()
    print "Proxy: job size for degridding = ", p.get_job_size_degridding()

    # allocate memory for data
    nr_polarizations = p.get_nr_polarizations()

    visibilities =  numpy.ones(
        (nr_baselines, nr_time, nr_channels, nr_polarizations),
        dtype = idg.visibilitiestype)

    uvw = numpy.zeros((nr_baselines, nr_time),
                      dtype = idg.uvwtype)
    idg.utils.init_uvw(uvw)
    # idg.utils.plot_uvw(uvw)

    wavenumbers = numpy.ones(nr_channels,
                             dtype = idg.wavenumberstype)
    metadata = numpy.zeros((nr_baselines, nr_timeslots),
                           dtype=idg.metadatatype)
    grid = numpy.zeros((nr_polarizations, grid_size, grid_size),
                       dtype = idg.gridtype)
    aterms = numpy.zeros((nr_stations, nr_timeslots, nr_polarizations,
                          subgrid_size, subgrid_size), \
                         dtype = idg.atermtype)
    # Set aterm to identity
    aterms[:,:,0,:,:] = 1.0
    aterms[:,:,3,:,:] = 1.0

    spheroidal = numpy.ones((subgrid_size, subgrid_size),
                            dtype = idg.spheroidaltype)

    # call gridding and degridding
    w_offset = 0.0

    p.grid_visibilities(visibilities, uvw, wavenumbers, metadata, grid,
                        w_offset, aterms, spheroidal)
    p.transform(idg.FourierDomainToImageDomain, grid)

    idg.utils.plot_grid(grid)

    p.transform(idg.ImageDomainToFourierDomain, grid)
    p.degrid_visibilities(visibilities, uvw, wavenumbers, metadata, grid,
                          w_offset, aterms, spheroidal)
