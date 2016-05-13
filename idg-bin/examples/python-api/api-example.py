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
_nr_time = 2*4096             # samples per baseline
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

# def init_aterms(nr_stations, nr_timeslots, subgrid_size):
#     aterms = numpy.zeros(
#         (nr_stations, nr_timeslots, _nr_polarizations, subgrid_size, subgrid_size),
#         dtype = idg.atermtype)
#     # idg.utils.init_aterms(aterms)

#     # TODO: update C++ init_aterms
#     # Set aterm to identity instead
#     aterms[:,:,0,:,:] = 1.0
#     aterms[:,:,3,:,:] = 1.0

#     return aterms

def init_spheroidal(subgrid_size):
    spheroidal = numpy.ones(
        (subgrid_size, subgrid_size),
        dtype = idg.spheroidaltype)
    idg.utils.init_spheroidal(spheroidal)
    #idg.utils.plot_spheroidal(spheroidal)
    return spheroidal


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
    grid = init_grid(grid_size)
    # aterms = init_aterms(nr_stations, nr_timeslots, subgrid_size)
    spheroidal = init_spheroidal(subgrid_size)

    frequencies = numpy.ndarray(nr_channels, dtype=numpy.float64)
    for i in range(nr_channels):
        frequencies[i] = sc.speed_of_light * wavenumbers[i] / (2*math.pi)

    ##################
    # initialize proxy
    ##################
    bufferTimesteps = 512

    plan = IDG.GridderPlan(bufferTimesteps)
    plan.set_stations(nr_stations);
    plan.set_frequencies(frequencies);
    plan.set_grid(grid);
    plan.set_subgrid_size(subgrid_size);
    plan.set_spheroidal(spheroidal);
    plan.set_image_size(0.1);
    plan.set_w_kernel_size(subgrid_size/2);
    plan.bake();

    ##########################
    # loop to fill buffer once
    ##########################

    fig, axarr = plt.subplots(2, 2)

    for time_major in range(nr_time / bufferTimesteps):
        for time_minor in range(bufferTimesteps):

            time = time_major*bufferTimesteps + time_minor

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
                    visibilities,
                    uvw_coordinates,
                    antenna1,
                    antenna2,
                    time
                )

        live_plot_grid(axarr, grid)

    ##############
    # flush buffer
    ##############
    plan.execute()


    plt.show()
