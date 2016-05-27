#!/usr/bin/env python

# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
import numpy
import matplotlib.pyplot as plt
import IDG
import idg   # HACK: utility to initialize data, TODO: rename


######
# main
######
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
    uvw           = idg.utils.get_example_uvw(nr_baselines, nr_time,
                                              integration_time)
    wavenumbers   = idg.utils.get_example_frequencies(nr_channels)
    baselines     = idg.utils.get_example_baselines(nr_baselines)

    frequencies   = idg.utils.get_example_frequencies(nr_channels,
                                                      dtype=numpy.float64)
    aterms        = idg.utils.get_example_aterms(1, nr_stations,
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
    idg.utils.plot_grid(grid_image)

    grid_gridded  = idg.utils.get_example_grid(nr_polarizations,
                                               grid_size,
                                               dtype = numpy.complex128)


    ######################################################################
    # Create plan
    ######################################################################
    bufferTimesteps = nr_time
    nr_timeslots = nr_time / bufferTimesteps

    degridder_plan = IDG.DegridderPlan(bufferTimesteps)
    degridder_plan.set_stations(nr_stations);
    degridder_plan.set_frequencies(frequencies);
    degridder_plan.set_spheroidal(spheroidal);
    degridder_plan.set_grid(grid_image);  # actually, give fft2(grid_image)
    degridder_plan.set_image_size(image_size);
    degridder_plan.set_w_kernel_size(subgrid_size/2);
    degridder_plan.internal_set_subgrid_size(subgrid_size);
    degridder_plan.bake();

    gridder_plan = IDG.GridderPlan(bufferTimesteps)
    gridder_plan.set_stations(nr_stations);
    gridder_plan.set_frequencies(frequencies);
    gridder_plan.set_spheroidal(spheroidal);
    gridder_plan.set_grid(grid_gridded);
    gridder_plan.set_image_size(image_size);
    gridder_plan.set_w_kernel_size(subgrid_size/2);
    gridder_plan.internal_set_subgrid_size(subgrid_size);
    gridder_plan.bake();

    # HACK: Should not be part of the plan
    # HACK: IDG.transform_grid(IDG.Direction.ImageToFourier, grid);
    # to be called before baking the plan
    degridder_plan.transform_grid(IDG.Direction.ImageToFourier, grid_image);

    ###########
    # Degridder
    ###########
    rowId = 0

    for time_batch in range(nr_time / bufferTimesteps):

        #### For each time chunk: set a-term, request visibilities
        degridder_plan.start_aterm(aterms[time_batch,:,:,:])

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
                degridder_plan.request_visibilities(
                    rowId,
                    uvw_coordinates,
                    antenna1,
                    antenna2,
                    time
                )
                rowId = rowId + 1

        degridder_plan.finish_aterm()  # has implicit flush
        degridder_plan.flush()

        #### For each time batch: set a-term, read the visibilities and grid them

        #gridder_plan.start_aterm(aterms[time_batch,:,:,:])

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

                visibilities = degridder_plan.read_visibilities(
                   antenna1,
                   antenna2,
                   time)
                # visibilities =  numpy.ones((nr_channels, nr_polarizations),
                #                            dtype=numpy.complex64)

                # Add visibilities to the buffer
                gridder_plan.grid_visibilities(
                    visibilities,
                    uvw_coordinates,
                    antenna1,
                    antenna2,
                    time
                )

        #gridder_plan.finish_aterm() # has implicit flush
        gridder_plan.flush()

        idg.utils.plot_grid(grid_gridded, scaling='log')

        gridder_plan.transform_grid(IDG.Direction.FourierToImage, grid_gridded);
        idg.utils.plot_grid(grid_gridded)

        plt.show()
