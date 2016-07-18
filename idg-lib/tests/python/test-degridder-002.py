#!/usr/bin/env python

# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
import idg
import numpy
import matplotlib.pyplot as plt


def visibilies_pt_src_w0(x, y, ampl, image_size, grid_size,
                         uvw, wavenumbers, vis):
    uvw['w'] = 0
    vis.fill(0)

    # create visibilities
    nr_baselines = vis.shape[0]
    nr_time = vis.shape[1]
    nr_channels = vis.shape[2]
    nr_polarizations = vis.shape[3]

    l = x*image_size/grid_size
    m = y*image_size/grid_size

    for b in range(nr_baselines):
        for t in range(nr_time):
            for c in range(nr_channels):
                u = wavenumbers[c]*uvw[b][t]['u']/(2*numpy.pi)
                v = wavenumbers[c]*uvw[b][t]['v']/(2*numpy.pi)
                value = ampl*numpy.exp(numpy.complex(0,-2*numpy.pi*(u*l + v*m)))
                for p in range(nr_polarizations):
                    vis[b][t][c][p] = value





if __name__ == "__main__":
    ############
    # paramaters
    ############
    nr_stations = 8
    nr_baselines = nr_stations*(nr_stations-1)/2
    nr_channels = 8
    nr_time = 4800            # samples per baseline
    nr_timeslots = 10         # A-term time slots
    image_size = 0.08
    subgrid_size = 32
    grid_size = 512
    integration_time = 10
    kernel_size = (subgrid_size / 2) + 1
    w_offset = 0

    ##################
    # initialize proxy
    ##################
    constructor = idg.CPU.Optimized
    p_cpu = constructor(nr_stations, nr_channels, nr_time, nr_timeslots, image_size, grid_size, subgrid_size)
    p = p_cpu

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
    #vis_true =  numpy.zeros(
    #    (nr_baselines, nr_time, nr_channels, nr_polarizations),
    #    dtype = idg.visibilitiestype)

    # uvw
    uvw = idg.utils.get_example_uvw(nr_baselines, nr_time, integration_time)
    #idg.utils.plot_uvw(uvw)

    # wavenumbers
    wavenumbers = idg.utils.get_example_wavenumbers(nr_channels)
    #idg.utils.plot_wavenumbers(wavenumbers)

    # baselines
    baselines = idg.utils.get_example_baselines(nr_baselines)

    # grid
    grid = idg.utils.get_example_grid(nr_polarizations, grid_size)

    # aterms
    aterms = idg.utils.get_example_aterms(nr_timeslots, nr_stations,
                                                 subgrid_size,
                                                 nr_polarizations)

    # aterm offset
    aterms_offset = idg.utils.get_example_aterms_offset(nr_timeslots,
                                                        nr_time)

    # spheroidal
    spheroidal = idg.utils.init_example_spheroidal_subgrid(subgrid_size)
    #idg.utils.plot_spheroidal(spheroidal)


    #
    # Add point source in the middle of the grid
    #
    offset_x = 80
    offset_y = 50
    amplitude = 2
    grid[:,grid_size/2+offset_y,grid_size/2+offset_x] = amplitude
    offset_x = 200
    offset_y = 120
    amplitude = 2
    grid[:,grid_size/2+offset_y,grid_size/2+offset_x] = amplitude
    offset_x = 30
    offset_y = 120
    amplitude = 5
    grid[2,grid_size/2-offset_y,grid_size/2-offset_x] = amplitude

    #visibilies_pt_src_w0(offset_x, offset_y, amplitude,
    #                     image_size, grid_size, uvw, wavenumbers, vis_true)
    #idg.utils.plot_visibilities(vis_true, form='abs')
    #idg.utils.plot_visibilities(vis_true, form='angle')


    ############
    # degridding
    ############
    p_cpu.transform(idg.ImageDomainToFourierDomain, grid)

    p_cpu.degrid_visibilities(visibilities, uvw, wavenumbers, baselines, grid,
                          w_offset, kernel_size, aterms, aterms_offset, spheroidal)
    idg.utils.plot_visibilities(visibilities, form='abs', maxtime=100)
    idg.utils.plot_visibilities(visibilities, form='angle', maxtime=100)

    # reset grid
    grid = numpy.zeros((nr_polarizations, grid_size, grid_size),
                       dtype = idg.gridtype)

    ##########
    # gridding
    ##########
    p_cpu.grid_visibilities(visibilities, uvw, wavenumbers, baselines, grid,
                        w_offset, kernel_size, aterms, aterms_offset, spheroidal)

    p_cpu.transform(idg.FourierDomainToImageDomain, grid)

    # plot result (should look like the point spread function)
    idg.utils.plot_grid(grid)

    plt.show()
