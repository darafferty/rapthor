import os
import math
import numpy
import ctypes
import numpy.ctypeslib
import matplotlib.pyplot as plt


# A bit ugly, but ctypes.util's find_library does not look in
# the LD_LIBRARY_PATH, but only PATH. Howver, we can also provide
# the full path of the shared object file
path = os.path.dirname(os.path.realpath(__file__))
path, junk = os.path.split(path)
path, junk = os.path.split(path)
libpath = os.path.join(path, 'libidg-utility.so')
lib = ctypes.cdll.LoadLibrary(libpath)


def nr_baselines_to_nr_stations(nr_baselines):
    lower = int(math.floor(math.sqrt(2*nr_baselines)))
    upper = int(math.ceil(math.sqrt(2*nr_baselines) + 2))
    nr_stations = 2;
    for i in range(lower, upper+1):
        if (i*(i-1)/2 == nr_baselines):
            nr_stations = i
            return nr_stations
    return nr_stations


def init_uvw(uvw):
    nr_baselines = uvw.shape[0]
    nr_stations = nr_baselines_to_nr_stations(nr_baselines)
    nr_time = uvw.shape[1]
    lib.utils_init_uvw( uvw.ctypes.data_as(ctypes.c_void_p),
                        nr_stations, nr_baselines, nr_time )


def plot_uvw(uvw):
    """Plot UVW data as (u,v)-plot
    Input: uvw - numpy.ndarray(shape=(nr_subgrids, nr_timesteps, 3),
                               dtype = idg.uvwtype)
    """
    u = uvw['u'].flatten()
    v = uvw['v'].flatten()
    uvlim = 1.2*max(max(abs(u)), max(abs(v)))
    plt.xlim([-uvlim, uvlim])
    plt.ylim([-uvlim, uvlim])
    plt.plot(numpy.append(u,-u),numpy.append(v,-v),'.')
    plt.grid(True)
    plt.axes().set_aspect('equal')
    plt.show()


def plot_grid(grid, form="abs"):
    """Plot Grid data
    Input: grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
                                dtype = idg.gridtype)
           form - "real", "imag", "abs", "phase"
    """
    f, axarr = plt.subplots(2, 2)
    if (form=="real"):
        gridXX = numpy.real(grid[0,:,:])
        gridXY = numpy.real(grid[1,:,:])
        gridYX = numpy.real(grid[2,:,:])
        gridYY = numpy.real(grid[3,:,:])
    elif (form=="imag"):
        gridXX = numpy.imag(grid[0,:,:])
        gridXY = numpy.imag(grid[1,:,:])
        gridYX = numpy.imag(grid[2,:,:])
        gridYY = numpy.imag(grid[3,:,:])
    elif (form=="angle"):
        gridXX = numpy.angle(grid[0,:,:])
        gridXY = numpy.angle(grid[1,:,:])
        gridYX = numpy.angle(grid[2,:,:])
        gridYY = numpy.angle(grid[3,:,:])
    else:
        gridXX = numpy.abs(grid[0,:,:])
        gridXY = numpy.abs(grid[1,:,:])
        gridYX = numpy.abs(grid[2,:,:])
        gridYY = numpy.abs(grid[3,:,:])

    axarr[0,0].imshow(gridXX)
    axarr[0,1].imshow(gridXY)
    axarr[1,0].imshow(gridYX)
    axarr[1,1].imshow(gridYY)

    axarr[0,0].set_title('XX')
    axarr[0,1].set_title('XY')
    axarr[1,0].set_title('YX')
    axarr[1,1].set_title('YY')

    axarr[0,0].tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        right='off',
        left='off',
        labelbottom='off',
        labelleft='off')

    axarr[0,1].tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        right='off',
        left='off',
        labelbottom='off',
        labelleft='off')

    axarr[1,0].tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        right='off',
        left='off',
        labelbottom='off',
        labelleft='off')

    axarr[1,1].tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        right='off',
        left='off',
        labelbottom='off',
        labelleft='off')

    plt.show()
