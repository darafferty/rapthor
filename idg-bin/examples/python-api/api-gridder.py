#!/usr/bin/env python

# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
import numpy
import matplotlib.pyplot as plt
import IDG
import idg   # HACK: utility to initialize data, TODO: rename


######################################################################
# Utility for plotting
######################################################################
def live_plot_grid(axarr, grid):
    gridXX = numpy.log(numpy.abs(grid[0,:,:]) + 1)
    gridXY = numpy.log(numpy.abs(grid[1,:,:]) + 1)
    gridYX = numpy.log(numpy.abs(grid[2,:,:]) + 1)
    gridYY = numpy.log(numpy.abs(grid[3,:,:]) + 1)

    axarr[0, 0].imshow(gridXX)
    axarr[0, 0].set_title('XX')
    axarr[0, 1].imshow(gridXY)
    axarr[0, 1].set_title('XY')
    axarr[1, 0].imshow(gridYX)
    axarr[1, 0].set_title('YX')
    axarr[1, 1].imshow(gridYY)
    axarr[1, 1].set_title('YY')

    # Hide ticks for plots
    plt.setp([a.get_xticklabels() for a in axarr[:,0]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:,0]], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr[:,1]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:,1]], visible=False)

    plt.pause(0.05)



######################################################################
# Main
######################################################################
if __name__ == "__main__":

    ######################################################################
    # Set parameters
    ######################################################################
    nr_stations = 22
    nr_baselines = nr_stations*(nr_stations-1)/2
    nr_channels = 1
    nr_time = 2048             # samples per baseline
    image_size = 0.20
    subgrid_size = 32
    grid_size = 512
    integration_time = 10
    kernel_size = (subgrid_size / 2) + 1
    nr_polarizations = 4


    ######################################################################
    # Print parameters
    ######################################################################
    print "nr_stations      = ", nr_stations
    print "nr_baselines     = ", nr_baselines
    print "nr_channels      = ", nr_channels
    print "nr_timesteps     = ", nr_time
    print "nr_polarizations = ", nr_polarizations
    print "subgrid_size     = ", subgrid_size
    print "grid_size        = ", grid_size
    print "image_size       = ", image_size
    print "kernel_size      = ", kernel_size


    ######################################################################
    # Initialize data
    ######################################################################
    bufferTimesteps = 512
    nr_timeslots = nr_time / bufferTimesteps

    uvw           = idg.utils.get_example_uvw(nr_baselines, nr_time,
                                              integration_time)
    wavenumbers   = idg.utils.get_example_frequencies(nr_channels)
    baselines     = idg.utils.get_example_baselines(nr_baselines)

    frequencies   = idg.utils.get_example_frequencies(nr_channels,
                                                      dtype=numpy.float64)
    aterms        = idg.utils.get_example_aterms(nr_timeslots, nr_stations,
                                                 subgrid_size,
                                                 nr_polarizations,
                                                 dtype = numpy.complex128)
    spheroidal    = idg.utils.get_example_spheroidal(subgrid_size,
                                                     dtype = numpy.float64)
    grid          = idg.utils.get_example_grid(nr_polarizations,
                                               grid_size,
                                               dtype = numpy.complex128)
    visibilities  = idg.utils.get_example_visibilities(nr_baselines,
                                                       nr_time,
                                                       nr_channels,
                                                       nr_polarizations,
                                                       image_size,
                                                       grid_size,
                                                       uvw,
                                                       wavenumbers)

    ######################################################################
    # Create plan
    ######################################################################
    plan = IDG.GridderPlan(IDG.Type.CPU_OPTIMIZED, bufferTimesteps)
    plan.set_stations(nr_stations);
    plan.set_frequencies(frequencies);
    plan.set_grid(grid);
    plan.set_spheroidal(spheroidal);
    plan.set_image_size(image_size);
    plan.set_w_kernel_size(subgrid_size/2);
    plan.internal_set_subgrid_size(subgrid_size);
    plan.bake();


    ######################################################################
    # Grid visibilities
    ######################################################################
    fig, axarr = plt.subplots(2, 2)

    for time_batch in range(nr_time / bufferTimesteps):

        plan.start_aterm(aterms[time_batch,:,:,:])

        for time_minor in range(bufferTimesteps):

            time = time_batch*bufferTimesteps + time_minor

            for bl in range(nr_baselines):

                # Set antenna indices (Note: smaller one first by convention of AO)
                antenna1 = baselines[bl][1]
                antenna2 = baselines[bl][0]

                # Set UVW coordinates in double precision
                uvw_coordinates = numpy.zeros(3, dtype=numpy.float64)
                uvw_coordinates[0] = uvw[bl][time]['u']
                uvw_coordinates[1] = uvw[bl][time]['v']
                uvw_coordinates[2] = uvw[bl][time]['w']

                # Set visibilities
                visibilities =  numpy.ones((nr_channels, nr_polarizations),
                                           dtype=numpy.complex64)

                # Add visibilities to the buffer
                plan.grid_visibilities(
                    time,
                    antenna1,
                    antenna2,
                    uvw_coordinates,
                    visibilities
                )

        plan.finish_aterm()

        live_plot_grid(axarr, grid)

    # Make sure buffer is flushed at the end
    plan.flush()

    plt.show()
