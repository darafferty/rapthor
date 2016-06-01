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
def live_plot_grid(axarr, grid, scaling='abs'):

    if (scaling=='log'):
        gridXX = numpy.log(numpy.abs(grid[0,:,:]) + 1)
        gridXY = numpy.log(numpy.abs(grid[1,:,:]) + 1)
        gridYX = numpy.log(numpy.abs(grid[2,:,:]) + 1)
        gridYY = numpy.log(numpy.abs(grid[3,:,:]) + 1)
    else:
        gridXX = numpy.abs(grid[0,:,:])
        gridXY = numpy.abs(grid[1,:,:])
        gridYX = numpy.abs(grid[2,:,:])
        gridYY = numpy.abs(grid[3,:,:])

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


######
# main
######
if __name__ == "__main__":

    ######################################################################
    # Set parameters
    ######################################################################
    nr_stations = 12
    nr_baselines = nr_stations*(nr_stations-1)/2
    nr_channels = 8
    nr_time = 2048             # samples per baseline
    image_size = 0.18
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
    bufferTimesteps = nr_time / 4
    nr_timeslots    = nr_time / bufferTimesteps

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
    visibilities  = idg.utils.get_example_visibilities(nr_baselines,
                                                       nr_time,
                                                       nr_channels,
                                                       nr_polarizations,
                                                       image_size,
                                                       grid_size,
                                                       uvw,
                                                       wavenumbers)

    grid_image    = idg.utils.get_example_grid(nr_polarizations,
                                               grid_size,
                                               dtype = numpy.complex128)
    # add point sources
    offset_x = 0
    offset_y = 0
    grid_image[:,(grid_size/2)+offset_y,(grid_size/2)+offset_x] = 1
    # idg.utils.plot_grid(grid_image)

    grid_gridded  = idg.utils.get_example_grid(nr_polarizations,
                                               grid_size,
                                               dtype = numpy.complex128)


    ######################################################################
    # Create plan
    ######################################################################
    degridder = IDG.DegridderPlan(IDG.Type.CPU_OPTIMIZED, bufferTimesteps)
    degridder.set_stations(nr_stations);
    degridder.set_frequencies(frequencies);
    degridder.set_spheroidal(spheroidal);
    degridder.set_grid(grid_image);  # actually, give fft2(grid_image)
    degridder.set_image_size(image_size);
    degridder.set_w_kernel_size(subgrid_size/2);
    degridder.internal_set_subgrid_size(subgrid_size);
    degridder.bake();

    gridder = IDG.GridderPlan(IDG.Type.CPU_OPTIMIZED, bufferTimesteps)
    gridder.set_stations(nr_stations);
    gridder.set_frequencies(frequencies);
    gridder.set_spheroidal(spheroidal);
    gridder.set_grid(grid_gridded);
    gridder.set_image_size(image_size);
    gridder.set_w_kernel_size(subgrid_size/2);
    gridder.internal_set_subgrid_size(subgrid_size);
    gridder.bake();

    # HACK: Should not be part of the plan
    # HACK: IDG.transform_grid(IDG.Direction.ImageToFourier, grid);
    # to be called before baking the plan
    degridder.transform_grid(IDG.Direction.ImageToFourier, grid_image);

    ###########
    # Degridder
    ###########
    fig, axarr = plt.subplots(2, 2)

    for time_batch in range(nr_time / bufferTimesteps):

        # For each time chunk: set a-term, request visibilities
        degridder.start_aterm(aterms[time_batch,:,:,:])

        for time_minor in range(bufferTimesteps):
            time = time_batch*bufferTimesteps + time_minor
            for bl in range(nr_baselines):

                # Set antenna indices (Note: smaller one first by convention)
                antenna1 = baselines[bl][1]
                antenna2 = baselines[bl][0]

                # Set UVW coordinates in double precision
                uvw_coordinates = numpy.zeros(3, dtype=numpy.float64)
                uvw_coordinates[0] = uvw[bl][time]['u']
                uvw_coordinates[1] = uvw[bl][time]['v']
                uvw_coordinates[2] = uvw[bl][time]['w']

                # Add visibilities to the buffer
                degridder.request_visibilities(
                    uvw_coordinates,
                    antenna1,
                    antenna2,
                    time
                )

        degridder.finish_aterm()

        # For each time batch: set a-term, read the visibilities and grid them
        gridder.start_aterm(aterms[time_batch,:,:,:])

        for time_minor in range(bufferTimesteps):

            time = time_batch*bufferTimesteps + time_minor
            for bl in range(nr_baselines):

                # Set antenna indices (Note: smaller one first by convention)
                antenna1 = baselines[bl][1]
                antenna2 = baselines[bl][0]

                # Set UVW coordinates in double precision
                uvw_coordinates = numpy.zeros(3, dtype=numpy.float64)
                uvw_coordinates[0] = uvw[bl][time]['u']
                uvw_coordinates[1] = uvw[bl][time]['v']
                uvw_coordinates[2] = uvw[bl][time]['w']

                visibilities = degridder.read_visibilities(
                   antenna1,
                   antenna2,
                   time)

                # Add visibilities to the buffer
                gridder.grid_visibilities(
                    visibilities,
                    uvw_coordinates,
                    antenna1,
                    antenna2,
                    time
                )

        gridder.finish_aterm()

        live_plot_grid(axarr, grid_gridded, scaling='log')

        grid_copy = numpy.copy(grid_gridded);
        gridder.transform_grid(IDG.Direction.FourierToImage, grid_copy)

        # live_plot_grid(axarr, grid_copy)

    plt.show()
