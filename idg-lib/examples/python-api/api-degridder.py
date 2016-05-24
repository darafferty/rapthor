#!/usr/bin/env python

# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
import numpy
import math
import scipy.constants as sc
import matplotlib.pyplot as plt
import IDG
import idg   # HACK: utility to initialize data, TODO: rename


############
# paramaters
############
_nr_stations = 22
_nr_baselines = _nr_stations*(_nr_stations-1)/2
_nr_channels = 1
_nr_time = 512             # samples per baseline
_image_size = 0.05
_subgrid_size = 24
_grid_size = 1024
_integration_time = 5
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

def init_uvw(nr_baselines, nr_time, integration_time):
    uvw = numpy.zeros((nr_baselines, nr_time),
                      dtype = idg.uvwtype)
    idg.utils.init_uvw(uvw, integration_time)
    # idg.utils.plot_uvw(uvw)
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
        dtype = numpy.complex128)
    return grid

def init_aterms(nr_timeslots, nr_stations, subgrid_size, nr_polarizations):
    aterms = numpy.zeros((nr_timeslots, nr_stations, subgrid_size,
                          subgrid_size, nr_polarizations),
                         dtype = numpy.complex128)
    aterms[:,:,:,:,0] = 1
    aterms[:,:,:,:,1] = 0
    aterms[:,:,:,:,2] = 0
    aterms[:,:,:,:,3] = 1
    return aterms

def init_spheroidal(subgrid_size):
    spheroidal = numpy.ones(
        (subgrid_size, subgrid_size),
        dtype = numpy.float64)
    idg.utils.init_spheroidal(spheroidal)
    #idg.utils.plot_spheroidal(spheroidal)
    return spheroidal


######
# main
######
if __name__ == "__main__":

    # Set parameters
    nr_stations = get_nr_stations()
    nr_baselines = get_nr_baselines()
    nr_channels = get_nr_channels()
    nr_time = get_nr_time()
    image_size = get_image_size()
    subgrid_size = get_subgrid_size()
    grid_size = get_grid_size()
    integration_time = get_integration_time()
    kernel_size = get_kernel_size()
    nr_polarizations = get_nr_polarizations()

    ##################
    # print parameters
    ##################
    print "nr_stations = ", nr_stations
    print "nr_baselines = ", nr_baselines
    print "nr_channels = ", nr_channels
    print "nr_timesteps = ", nr_time
    print "nr_polarizations = ", nr_polarizations
    print "subgrid_size = ", subgrid_size
    print "grid_size = ", grid_size
    print "image_size = ", image_size
    print "kernel_size = ", kernel_size

    #################
    # initialize data
    #################
    uvw = init_uvw(nr_baselines, nr_time, integration_time)
    wavenumbers = init_wavenumbers(nr_channels)
    baselines = init_baselines(nr_baselines)
    spheroidal = init_spheroidal(subgrid_size)

    grid_image = init_grid(grid_size)
    grid_image[:,grid_size/2,grid_size/2] = 1
    # idg.utils.plot_grid(grid_image)

    grid_gridded = init_grid(grid_size)

    frequencies = numpy.ndarray(nr_channels, dtype=numpy.float64)
    for i in range(nr_channels):
        frequencies[i] = sc.speed_of_light * wavenumbers[i] / (2*math.pi)

    bufferTimesteps = nr_time
    nr_timeslots = nr_time / bufferTimesteps
    aterms = init_aterms(nr_timeslots, nr_stations, subgrid_size, 4)

    ##################
    # initialize proxy
    ##################

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

    for time_major in range(nr_time / bufferTimesteps):

        # #### For each time chunk: set a-term, request visibilities
        # degridder_plan.start_aterm(aterms[time_major,:,:,:])

        # for time_minor in range(bufferTimesteps):
        #     time = time_major*bufferTimesteps + time_minor
        #     for bl in range(nr_baselines):

        #         # Set antenna indices (Note: smaller one first by convention)
        #         antenna1 = baselines[bl][1]
        #         antenna2 = baselines[bl][0]

        #         # Set UVW coordinates in double precision
        #         uvw_coordinates = numpy.zeros(3, dtype=numpy.float64)
        #         uvw_coordinates[0] = uvw[bl][time]['u']
        #         uvw_coordinates[1] = uvw[bl][time]['v']
        #         uvw_coordinates[2] = uvw[bl][time]['w']

        #         # Add visibilities to the buffer
        #         degridder_plan.request_visibilities(
        #             rowId,
        #             uvw_coordinates,
        #             antenna1,
        #             antenna2,
        #             time
        #         )
        #         rowId = rowId + 1

        # degridder_plan.finish_aterm()  # has implicit flush
        # degridder_plan.flush()

        #### For each time chunk: set a-term, read the visibilities and grid them

        #gridder_plan.start_aterm(aterms[time_major,:,:,:])

        for time_minor in range(bufferTimesteps):
            time = time_major*bufferTimesteps + time_minor
            for bl in range(nr_baselines):

                # Set antenna indices (Note: smaller one first by convention)
                antenna1 = baselines[bl][1]
                antenna2 = baselines[bl][0]

                # Set UVW coordinates in double precision
                uvw_coordinates = numpy.zeros(3, dtype=numpy.float64)
                uvw_coordinates[0] = uvw[bl][time]['u']
                uvw_coordinates[1] = uvw[bl][time]['v']
                uvw_coordinates[2] = uvw[bl][time]['w']

                #visibilities = degridder_plan.read_visibilities(
                #    antenna1,
                #    antenna2,
                #    time)
                visibilities =  numpy.ones((nr_channels, nr_polarizations),
                                           dtype=numpy.complex64)

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

        grid_test = numpy.fft.fftshift(grid_gridded, axes=[1, 2])
        grid_test = numpy.fft.fft2(grid_test, axes=[1, 2])
        grid_test = numpy.fft.fftshift(grid_test, axes=[1, 2])

        gridder_plan.transform_grid(IDG.Direction.FourierToImage, grid_gridded);
        idg.utils.plot_grid(grid_gridded, scaling='log')
        idg.utils.plot_grid(grid_test, scaling='log')
        idg.utils.plot_grid(grid_test)

        plt.show()
