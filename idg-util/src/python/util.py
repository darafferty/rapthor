import os
import math
import numpy
import ctypes
import numpy.ctypeslib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import scipy.constants as sc

from idgtypes import *

# Load idg-util library
lib = ctypes.cdll.LoadLibrary('libidg-util.so')


def resize_spheroidal(spheroidal, size, dtype=numpy.float32):
    subgrid_size = spheroidal.shape[0]  # assumes squares spheroidal
    tmp = spheroidal.astype(numpy.float32)
    result = numpy.zeros(shape=(size, size),
                         dtype=numpy.float32)
    lib.utils_resize_spheroidal( tmp.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_int(subgrid_size),
                                 result.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_int(size) )
    return result.astype(dtype)



def nr_baselines_to_nr_stations(nr_baselines):
    """Convert NUMBER OF BASELINES to NUMBER OF STATIONS"""
    lower = int(math.floor(math.sqrt(2*nr_baselines)))
    upper = int(math.ceil(math.sqrt(2*nr_baselines) + 2))
    nr_stations = 2;
    for i in range(lower, upper+1):
        if (i*(i-1)/2 == nr_baselines):
            nr_stations = i
            return nr_stations
    return nr_stations


def add_pt_src(
    x, y, amplitude,
    nr_baselines, nr_time, nr_channels, nr_polarizations,
    image_size, grid_size,
    uvw, frequencies, vis):

    lib.utils_add_pt_src.argtypes = [
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p]

    lib.utils_add_pt_src(
        ctypes.c_float(x),
        ctypes.c_float(y),
        ctypes.c_float(amplitude),
        ctypes.c_int(nr_baselines),
        ctypes.c_int(nr_time),
        ctypes.c_int(nr_channels),
        ctypes.c_int(nr_polarizations),
        ctypes.c_float(image_size),
        ctypes.c_int(grid_size),
        uvw.ctypes.data_as(ctypes.c_void_p),
        frequencies.ctypes.data_as(ctypes.c_void_p),
        vis.ctypes.data_as(ctypes.c_void_p))


def func_spheroidal(nu):
    """Function to compute spheroidal
        Based on reference code by Bas"""
    P = numpy.array([[ 8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1,  2.312756e-1],
                [ 4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2]])
    Q = numpy.array([[1.0000000e0, 8.212018e-1, 2.078043e-1],
                [1.0000000e0, 9.599102e-1, 2.918724e-1]])

    part = 0;
    end = 0.0;

    if (nu >= 0.0 and nu < 0.75):
        part = 0
        end = 0.75
    elif (nu >= 0.75 and nu <= 1.00):
        part = 1
        end = 1.00
    else:
        return 0.0

    nusq = nu * nu
    delnusq = nusq - end * end
    delnusqPow = delnusq
    top = P[part][0]
    for k in range(1,5):
        top += P[part][k] * delnusqPow
        delnusqPow *= delnusq

    bot = Q[part][0]
    delnusqPow = delnusq
    for k in range(1,3):
        bot += Q[part][k] * delnusqPow
        delnusqPow *= delnusq

    if bot == 0:
        result = 0
    else:
        result = (1.0 - nusq) * (top / bot)
    return result


def make_gaussian(size, fwhm = 3, center=None):
    x = numpy.arange(0, size, 1, float)
    y = x[:,numpy.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return numpy.exp(-4*numpy.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def init_example_spheroidal_subgrid(subgrid_size):
    """Construct spheroidal for subgrid"""
    # Spheroidal from Bas
    x = numpy.array(numpy.abs(numpy.linspace(-1, 1, num=subgrid_size, endpoint=False)), dtype=numpy.float32)
    x = numpy.array(map(lambda e: func_spheroidal(e), x), dtype=numpy.float32)
    spheroidal = x[numpy.newaxis,:] * x[:, numpy.newaxis]
    return spheroidal
    # Ones
    #return numpy.ones((subgrid_size, subgrid_size), dtype = numpy.float32)
    # Gaussian
    #return make_gaussian(subgrid_size, int(subgrid_size * 0.3))


def init_example_spheroidal_grid(subgrid_size, grid_size):
    """Construct spheroidal for grid"""
    spheroidal = init_example_spheroidal_subgrid(subgrid_size)
    s = numpy.fft.fft2(spheroidal)
    s = numpy.fft.fftshift(s)
    s1 = numpy.zeros((grid_size, grid_size), dtype = numpy.complex64)
    support_size1 = int((grid_size - subgrid_size)/2)
    support_size2 = int((grid_size + subgrid_size)/2)
    s1[support_size1:support_size2, support_size1:support_size2] = s
    s1 = numpy.fft.ifftshift(s1)
    return numpy.real(numpy.fft.ifft2(s1))


def init_grid_of_point_sources(N, image_size, visibilities, uvw,
                               frequencies, asymmetric=False):
    """Initialize visibilities (and set w=0) to
    get a grid of N by N point sources

    Arguments:
    N - odd integer for N by N point sources
    image_size - ...
    visibilities - numpy.ndarray(shape=(nr_baselines, nr_time,
                                 nr_channels, nr_polarizations),
                                 dtype=idg.visibilitiestype)
    uvw - numpy.ndarray(shape=(nr_baselines,nr_time),
                        dtype = idg.uvwtype)
    frequencies - numpy.ndarray(nr_channels, dtype = idg.frequenciestype)
    asymmetric - bool to make positive (l,m) twice in magnitude
    """

    # make sure N is odd, w=0, visibilities are zero initially
    if math.fmod(N,2)==0:
        N += 1
    uvw['w'] = 0
    visibilities.fill(0)

    # create visibilities
    nr_baselines     = visibilities.shape[0]
    nr_time          = visibilities.shape[1]
    nr_channels      = visibilities.shape[2]
    nr_polarizations = visibilities.shape[3]

    for b in range(nr_baselines):
        for t in range(nr_time):
            for c in range(nr_channels):
                u = frequencies[c]*uvw[b][t]['u']/(sc.speed_of_light)
                v = frequencies[c]*uvw[b][t]['v']/(sc.speed_of_light)
                for i in range(-N/2+1,N/2+1):     # -N/2,-N/2+1,..,-1,0,1,...,N/2
                    for j in range(-N/2+1,N/2+1): # -N/2,-N/2+1,..,-1,0,1,...,N/2
                        l = i*image_size/(N+1)
                        m = j*image_size/(N+1)
                        value = numpy.exp(numpy.complex(0,-2*numpy.pi*(u*l + v*m)))
                        if asymmetric==True:
                            if l>0 and m>0:
                                value *= 2
                        for p in range(nr_polarizations):
                            visibilities[b][t][c][p] += value


##### BEGIN: PLOTTING UTILITY       #####

def get_figure_name(name):
    return "Figure %d: %s" % (len(plt.get_fignums()) + 1, name)


def plot_uvw(uvw):
    """Plot UVW data as (u,v)-plot
    Input:
    uvw - numpy.ndarray(shape=(nr_subgrids, nr_timesteps, 3),
                        dtype = idg.uvwtype)
    """
    u = uvw['u'].flatten()
    v = uvw['v'].flatten()
    uvlim = 1.2*max(max(abs(u)), max(abs(v)))
    fig = plt.figure(get_figure_name("uvw"))
    plt.plot(numpy.append(u,-u),numpy.append(v,-v),'.')
    plt.xlim([-uvlim, uvlim])
    plt.ylim([-uvlim, uvlim])
    plt.grid(True)
    plt.axes().set_aspect('equal')

def plot_uvw_pixels(uvw, frequencies, image_size):
    """Plot UVW data as (u,v)-plot, scaled to pixel coordinates
    Input:
    uvw - numpy.ndarray(shape=(nr_subgrids, nr_timesteps, 3),
                        dtype = idg.uvwtype)
    """
    speed_of_light = 299792458.0
    u_ = numpy.array([], dtype = numpy.float32)
    v_ = numpy.array([], dtype = numpy.float32)
    for frequency in frequencies:
        u_ = numpy.append(u_, uvw['u'].flatten() * image_size * (frequency / speed_of_light))
        v_ = numpy.append(v_, uvw['v'].flatten() * image_size * (frequency / speed_of_light))
    uvlim = 1.2*max(max(abs(u_)), max(abs(v_)))
    fig = plt.figure(get_figure_name("uvw"))
    plt.plot(u_, v_,'.')
    plt.xlim([-uvlim, uvlim])
    plt.ylim([-uvlim, uvlim])
    plt.grid(True)
    plt.axes().set_aspect('equal')

def plot_uvw_meters(uvw):
    """Plot UVW data as (u,v)-plot
    Input:
    uvw - numpy.ndarray(shape=(nr_subgrids, nr_timesteps, 3),
                        dtype = idg.uvwtype)
    """
    u = uvw['u'].flatten()
    v = uvw['v'].flatten()
    uvlim = 1.2*max(max(abs(u)), max(abs(v)))
    fig = plt.figure(get_figure_name("uvw"))
    plt.plot(u, v,'.')
    plt.xlim([-uvlim, uvlim])
    plt.ylim([-uvlim, uvlim])
    plt.grid(True)
    plt.axes().set_aspect('equal')

def output_uvw(uvw):
    """Plot UVW data as (u,v)-plot to high-resolution png file
    Input:
    uvw - numpy.ndarray(shape=(nr_subgrids, nr_timesteps, 3),
                        dtype = idg.uvwtype)
    """
    u = uvw['u'].flatten()
    v = uvw['v'].flatten()
    fig = plt.figure(figsize=(40,40), dpi=300)
    plt.plot(numpy.append(u,-u),numpy.append(v,-v),'.', color='black', alpha=0.8, markersize=1.0)
    plt.axes().set_aspect('equal')
    plt.axis('off')
    plt.savefig("uvw-coverage.png")

def plot_frequencies(frequencies):
    """Plot frequencies
    Input:
    frequencies - numpy.ndarray(nr_channels, dtype = idg.frequenciestype)
    """
    fig = plt.figure(get_figure_name("frequencies"))
    plt.plot(frequencies,'.')
    plt.grid(True)
    plt.xlabel("Channel")
    plt.ylabel("rad/m")


def plot_visibilities(visibilities, form='abs', maxtime=numpy.inf):
    """Plot Grid data
    Input:
    visibilities - numpy.ndarray(shape=(nr_baselines, nr_time,
                                nr_channels, nr_polarizations),
                                dtype=idg.visibilitiestype)
    form - 'real', 'imag', 'abs', 'angle'
    """

    if maxtime>visibilities.shape[1]:
        maxtime=visibilities.shape[1]+1

    if (form=='real'):
        visXX = numpy.real( visibilities[:,:maxtime,:,0].flatten() )
        visXY = numpy.real( visibilities[:,:maxtime,:,1].flatten() )
        visYX = numpy.real( visibilities[:,:maxtime,:,2].flatten() )
        visYY = numpy.real( visibilities[:,:maxtime,:,3].flatten() )
        title = 'Real'
    elif (form=='imag'):
        visXX = numpy.imag( visibilities[:,:maxtime,:,0].flatten() )
        visXY = numpy.imag( visibilities[:,:maxtime,:,1].flatten() )
        visYX = numpy.imag( visibilities[:,:maxtime,:,2].flatten() )
        visYY = numpy.imag( visibilities[:,:maxtime,:,3].flatten() )
        title = 'Imag'
    elif (form=='angle'):
        visXX = numpy.angle( visibilities[:,:maxtime,:,0].flatten() )
        visXY = numpy.angle( visibilities[:,:maxtime,:,1].flatten() )
        visYX = numpy.angle( visibilities[:,:maxtime,:,2].flatten() )
        visYY = numpy.angle( visibilities[:,:maxtime,:,3].flatten() )
        title = 'Angle'
    else:
        visXX = numpy.abs( visibilities[:,:maxtime,:,0].flatten() )
        visXY = numpy.abs( visibilities[:,:maxtime,:,1].flatten() )
        visYX = numpy.abs( visibilities[:,:maxtime,:,2].flatten() )
        visYY = numpy.abs( visibilities[:,:maxtime,:,3].flatten() )
        title = 'Abs'

    fig, axarr = plt.subplots(2, 2, num=get_figure_name("visibilities"))
    fig.suptitle(title, fontsize=14)

    axarr[0,0].plot(visXX)
    axarr[0,1].plot(visXY)
    axarr[1,0].plot(visYX)
    axarr[1,1].plot(visYY)

    axarr[0,0].set_title('XX')
    axarr[0,1].set_title('XY')
    axarr[1,0].set_title('YX')
    axarr[1,1].set_title('YY')

    axarr[0,0].tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')

    axarr[0,1].tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')

    axarr[1,0].tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')

    axarr[1,1].tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')


def plot_aterms(aterms):
    """Plot A-terms
    Input:
    aterms - numpy.ndarray(shape=(nr_timeslots, nr_stations,
                           subgrid_size, subgrid_size, nr_polarizations),
                           dtype = idg.atermtype)
    """
    print "TO BE IMPLEMENTED"


def plot_spheroidal(spheroidal, interpolation_method='none'):
    """Plot spheroidal
    Input:
    spheroidal - numpy.ndarray(shape=(subgrid_size, subgrid_size),
                               dtype = idg.spheroidaltype)
    interpolation_method - 'none', 'nearest', 'bilinear', 'bicubic',
                           'spline16', ... (see matplotlib imshow)
    """
    plt.figure(get_figure_name("spheroidal"))
    plt.imshow(spheroidal, interpolation=interpolation_method)
    plt.colorbar()


def plot_grid_all(grid, form='abs', scaling='none', interpolation_method='none'):
    """Plot Grid data
    Input:
    grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
                         dtype = idg.gridtype)
    form - 'real', 'imag', 'abs', 'angle'
    scaling - 'none', 'log', 'sqrt'
    interpolation_method - 'none', 'nearest', 'bilinear', 'bicubic',
                           'spline16', ... (see matplotlib imshow)
    """
    if (scaling=='log'):
        grid = numpy.abs(grid) + 1
        grid = numpy.log(grid)
    if (scaling=='sqrt'):
        grid = numpy.sqrt(grid)

    if (form=='real'):
        gridXX = numpy.real(grid[0,:,:])
        gridXY = numpy.real(grid[1,:,:])
        gridYX = numpy.real(grid[2,:,:])
        gridYY = numpy.real(grid[3,:,:])
        title = 'Real'
    elif (form=='imag'):
        gridXX = numpy.imag(grid[0,:,:])
        gridXY = numpy.imag(grid[1,:,:])
        gridYX = numpy.imag(grid[2,:,:])
        gridYY = numpy.imag(grid[3,:,:])
        title = 'Imag'
    elif (form=='angle'):
        gridXX = numpy.angle(grid[0,:,:])
        gridXY = numpy.angle(grid[1,:,:])
        gridYX = numpy.angle(grid[2,:,:])
        gridYY = numpy.angle(grid[3,:,:])
        title = 'Angle'
    else:
        gridXX = numpy.abs(grid[0,:,:])
        gridXY = numpy.abs(grid[1,:,:])
        gridYX = numpy.abs(grid[2,:,:])
        gridYY = numpy.abs(grid[3,:,:])
        title = 'Abs'

    fig = plt.figure(get_figure_name("grid"))
    fig.suptitle(title, fontsize=14)

    ax = ["ax1", "ax2", "ax3", "ax4"]
    for idx in range(len(ax)):
        locals()[ax[idx]] = fig.add_subplot(2, 2, (idx + 1))
        divider = make_axes_locatable(vars()[ax[idx]])
        locals()["c" + ax[idx]] = divider.append_axes("right", size = "5%", pad = 0.05)

    im1 = locals()['ax1'].imshow(gridXX, interpolation=interpolation_method)
    plt.colorbar(im1, cax = locals()['cax1'], format='%.1e')

    im2 = locals()['ax2'].imshow(gridXY, interpolation=interpolation_method)
    plt.colorbar(im2, cax = locals()['cax2'], format='%.1e')

    im3 = locals()['ax3'].imshow(gridYX, interpolation=interpolation_method)
    plt.colorbar(im3, cax = locals()['cax3'], format='%.1e')

    im4 = locals()['ax4'].imshow(gridYY, interpolation=interpolation_method)
    plt.colorbar(im4, cax = locals()['cax4'], format='%.1e')

    locals()['ax1'].set_title('XX')
    locals()['ax2'].set_title('XY')
    locals()['ax3'].set_title('YX')
    locals()['ax4'].set_title('YY')

    locals()['ax1'].tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        right='off',
        left='off',
        labelbottom='off',
        labelleft='off')

    locals()['ax2'].tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        right='off',
        left='off',
        labelbottom='off',
        labelleft='off')

    locals()['ax3'].tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        right='off',
        left='off',
        labelbottom='off',
        labelleft='off')

    locals()['ax4'].tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        right='off',
        left='off',
        labelbottom='off',
        labelleft='off')


def plot_grid(grid, form='abs', scaling='none', interpolation_method='none', pol='all'):
    """Plot Grid data
    Input:
    grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
                         dtype = idg.gridtype)
    form - 'real', 'imag', 'abs', 'angle'
    scaling - 'none', 'log', 'sqrt'
    interpolation_method - 'none', 'nearest', 'bilinear', 'bicubic',
                           'spline16', ... (see matplotlib imshow)
    """
    if (pol=='all'):
        plot_grid_all(grid, form, scaling, interpolation_method)
        return

    if (scaling=='log'):
        grid = numpy.abs(grid) + 1
        grid = numpy.log(grid)
    if (scaling=='sqrt'):
        grid = numpy.sqrt(grid)

    if (form=='real'):
        grid = numpy.real(grid[pol,:,:])
        title = 'Real'
    elif (form=='imag'):
        grid = numpy.imag(grid[pol,:,:])
        title = 'Imag'
    elif (form=='angle'):
        grid = numpy.angle(grid[pol,:,:])
        title = 'Angle'
    else:
        grid = numpy.abs(grid[pol,:,:])
        title = 'Abs'

    fig = plt.figure(get_figure_name("grid"))
    fig.suptitle(title, fontsize=14)

    plt.imshow(grid, interpolation=interpolation_method)
    plt.colorbar(format='%.1e')

    plt.title(["XX", "XY", "YX", "YY"][pol])

    plt.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        right='off',
        left='off',
        labelbottom='off',
        labelleft='off')


def plot_metadata(metadata, uvw, frequencies, grid_size, subgrid_size, image_size):
    # Show subgrids (from metadata)
    x = metadata['coordinate']['x'].flatten()
    y = metadata['coordinate']['y'].flatten()
    fig = plt.figure(get_figure_name("metadata: {} subgrids".format(len(x))))
    grid = numpy.zeros((grid_size, grid_size))
    for coordinate in zip(x, y):
        _x = coordinate[0]
        _y = coordinate[1]
        grid[_y:_y+subgrid_size,_x:_x+subgrid_size] += 1
    grid[grid == 0] = numpy.nan
    plt.imshow(grid, interpolation='None')

    # Show u,v coordinates (from uvw)
    u = uvw['u'].flatten()
    v = uvw['v'].flatten()
    u_pixels = []
    v_pixels = []
    for frequency in frequencies:
        scaling = frequency * image_size / sc.speed_of_light
        u_pixels.append(u * scaling)
        v_pixels.append(v * scaling)
    u_pixels = numpy.asarray(u_pixels).flatten() + (grid_size / 2)
    v_pixels = numpy.asarray(v_pixels).flatten() + (grid_size / 2)
    #plt.plot(u_pixels, v_pixels, 'r.', markersize=2, alpha=0.9)

    # Make mouseover show value of grid
    def format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        z = grid[row,col]
        if z is not numpy.nan:
            return 'x=%1.1f, y=%1.1f, z=%1.1f' % (x, y, z)
        else:
            return 'x=%1.1f, y=%1.1f' % (x, y)

    ax = fig.gca()
    ax.format_coord = format_coord

    # Set plot options
    plt.grid(True)
    plt.colorbar()
    plt.axes().set_aspect('equal')
    plt.xlim([0, grid_size])
    plt.ylim([grid_size, 0])


##### END:   PLOTTING UTILITY       #####

##### BEGIN: INITIALZE DATA         #####

def init_identity_aterms(aterms):
    """Initialize aterms for test case defined in utility/initialize"""
    nr_timeslots     = aterms.shape[0]
    nr_stations      = aterms.shape[1]
    subgrid_size     = aterms.shape[2]
    nr_polarizations = aterms.shape[4]
    lib.utils_init_identity_aterms.argtypes = [ctypes.c_void_p,
                                               ctypes.c_int,
                                               ctypes.c_int,
                                               ctypes.c_int,
                                               ctypes.c_int]
    lib.utils_init_identity_aterms(aterms.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_int(nr_timeslots),
                                   ctypes.c_int(nr_stations),
                                   ctypes.c_int(subgrid_size),
                                   ctypes.c_int(nr_polarizations))


def init_identity_spheroidal(spheroidal):
    subgrid_size = spheroidal.shape[0]
    lib.utils_init_identity_spheroidal.argtypes = [ctypes.c_void_p,
                                                   ctypes.c_int]
    lib.utils_init_identity_spheroidal(spheroidal.ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_int(subgrid_size))


def get_identity_aterms(nr_timeslots, nr_stations, subgrid_size, nr_polarizations,
                        dtype=atermtype, info=False):
    aterms = numpy.zeros(
        (nr_timeslots, nr_stations, subgrid_size, subgrid_size, nr_polarizations),
        dtype = atermtype)
    init_identity_aterms(aterms)
    if info==True:
        print "aterms: numpy.ndarray(shape = (nr_timeslots, nr_stations," + \
              "subgrid_size, subgrid_size, nr_polarizations), " + \
              "dtype = " + str(dtype) + ")"
    return aterms.astype(dtype=dtype)


def get_zero_grid(nr_polarizations, grid_size,
                  dtype=gridtype, info=False):
    grid = numpy.zeros((nr_polarizations, grid_size, grid_size),
                       dtype=dtype)
    if info==True:
        print "grid: numpy.ndarray(shape = (nr_polarizations, grid_size, grid_size), " + \
                                   "dtype = " + str(dtype) + ")"
    return grid


def get_identity_spheroidal(subgrid_size, dtype=spheroidaltype, info=False):
    spheroidal = numpy.zeros(shape=(subgrid_size, subgrid_size),
                             dtype=spheroidaltype)
    init_identity_spheroidal(spheroidal)
    if info==True:
        print "grid: numpy.ndarray(shape = (subgrid_size, subgrid_size), " + \
                                   "dtype = " + str(dtype) + ")"
    return spheroidal.astype(dtype=dtype)


def get_zero_visibilities(nr_baselines, nr_time, nr_channels, nr_polarizations,
                          dtype=visibilitiestype, info=False):
    visibilities = numpy.zeros(shape=(nr_baselines, nr_time, nr_channels, nr_polarizations),
                               dtype=visibilitiestype)
    return visibilities.astype(dtype=dtype)


##### END:   INITIALZE DATA         #####

##### BEGIN: INITIALZE EXAMPLE DATA #####

def init_example_uvw(uvw, integration_time = 10):
    """Initialize uvw for test case defined in utility/initialize"""
    nr_baselines = uvw.shape[0]
    nr_stations  = nr_baselines_to_nr_stations(nr_baselines)
    nr_time      = uvw.shape[1]
    lib.utils_init_example_uvw.argtypes = [ctypes.c_void_p,
                                           ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_int,
                                           ctypes.c_float]
    lib.utils_init_example_uvw( uvw.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(nr_stations),
                                ctypes.c_int(nr_baselines),
                                ctypes.c_int(nr_time),
                                ctypes.c_float(integration_time))


def init_example_frequencies(frequencies):
    """Initialize frequencies for test case defined in utility/initialize"""
    nr_channels = frequencies.shape[0]
    lib.utils_init_example_frequencies.argtypes = [ctypes.c_void_p,
                                                   ctypes.c_int]
    lib.utils_init_example_frequencies(frequencies.ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_int(nr_channels) )


def init_dummy_visibilities(visibilities):
    """Initialize visibilities for test case defined in utility/initialize"""
    nr_baselines     = visibilities.shape[0]
    nr_time          = visibilities.shape[1]
    nr_channels      = visibilities.shape[2]
    nr_polarizations = visibilities.shape[3]
    lib.utils_init_example_visibilities.argtypes = [ctypes.c_void_p,
                                                    ctypes.c_int,
                                                    ctypes.c_int,
                                                    ctypes.c_int,
                                                    ctypes.c_int]
    lib.utils_init_dummy_visibilities(visibilities.ctypes.data_as(ctypes.c_void_p),
                                        ctypes.c_int(nr_baselines),
                                        ctypes.c_int(nr_time),
                                        ctypes.c_int(nr_channels),
                                        ctypes.c_int(nr_polarizations) )


def init_identity_aterms(aterms):
    """Initialize aterms for test case defined in utility/initialize"""
    nr_timeslots     = aterms.shape[0]
    nr_stations      = aterms.shape[1]
    subgrid_size     = aterms.shape[2]
    nr_polarizations = aterms.shape[4]
    lib.utils_init_identity_aterms.argtypes = [ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int]
    lib.utils_init_identity_aterms(aterms.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(nr_timeslots),
                                  ctypes.c_int(nr_stations),
                                  ctypes.c_int(subgrid_size),
                                  ctypes.c_int(nr_polarizations))


def init_example_spheroidal(spheroidal):
    """Initialize spheroidal for test case defined in utility/initialize"""
    subgrid_size = spheroidal.shape[0]
    lib.utils_init_example_spheroidal.argtypes = [ctypes.c_void_p,
                                                  ctypes.c_int]
    lib.utils_init_example_spheroidal(spheroidal.ctypes.data_as(ctypes.c_void_p),
                                      ctypes.c_int(subgrid_size) )

def init_example_aterms(aterms, nr_timeslots, nr_stations, height, width):
    """Initialize aterms"""
    lib.utils_init_example_aterms_offset.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_int]
    lib.utils_init_example_aterms(aterms.ctypes.data_as(ctypes.c_void_p),
                                         ctypes.c_int(nr_timeslots),
                                         ctypes.c_int(nr_stations),
                                         ctypes.c_int(height),
                                         ctypes.c_int(width))


def init_example_aterms_offset(aterms_offset, nr_time):
    """Initialize aterms offset"""
    nr_timeslots = aterms_offset.shape[0] - 1
    lib.utils_init_example_aterms_offset.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_int]
    lib.utils_init_example_aterms_offset(aterms_offset.ctypes.data_as(ctypes.c_void_p),
                                         ctypes.c_int(nr_timeslots),
                                         ctypes.c_int(nr_time))


def init_example_baselines(baselines):
    """Initialize baselines
    Input:
    baselines - numpy.ndarray(shape=(nr_baselines), dtype = idg.baselinetype)
    """
    nr_baselines = baselines.shape[0]
    nr_stations = nr_baselines_to_nr_stations(nr_baselines)
    lib.utils_init_example_baselines.argtypes = [ctypes.c_void_p,
                                                 ctypes.c_int,
                                                 ctypes.c_int]
    lib.utils_init_example_baselines(baselines.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_int(nr_stations),
                                     ctypes.c_int(nr_baselines))


def get_example_uvw(nr_baselines, nr_time, integration_time,
                    dtype=uvwtype, info=False):
    """Initialize and return example UVW array"""
    uvw = numpy.zeros((nr_baselines, nr_time),
                      dtype=uvwtype)
    init_example_uvw(uvw, integration_time)
    if info==True:
        print "uvw: numpy.ndarray(shape = (nr_baselines, nr_time), " + \
                                 "dtype = " + str(dtype) + ")"
    return uvw.astype(dtype=dtype)


def get_example_frequencies(nr_channels,
                            dtype=frequenciestype, info=False):
    """Initialize and returns example frequencies array"""
    frequencies = numpy.ones(nr_channels,
                             dtype=frequenciestype)
    init_example_frequencies(frequencies)
    if info==True:
        print "frequencies: numpy.ndarray(shape = (nr_channels), " + \
                                          "dtype = " + str(dtype) + ")"
    return frequencies.astype(dtype=dtype)


def get_example_baselines(nr_baselines,
                          dtype=baselinetype, info=False):
    """Initialize and return example baselines array"""
    baselines = numpy.zeros(nr_baselines,
                            dtype = baselinetype)
    init_example_baselines(baselines)
    if info==True:
        print "baselines: numpy.ndarray(shape = (nr_channels), " + \
                                        "dtype = " + str(dtype) + ")"
    return baselines.astype(dtype=dtype)


def get_example_grid(nr_polarizations, grid_size,
                     dtype=gridtype, info=False):
    grid = numpy.zeros((nr_polarizations, grid_size, grid_size),
                       dtype=dtype)
    if info==True:
        print "grid: numpy.ndarray(shape = (nr_polarizations, grid_size, grid_size), " + \
                                   "dtype = " + str(dtype) + ")"
    return grid


def get_example_aterms(nr_timeslots, nr_stations, subgrid_size, nr_polarizations,
                       dtype=atermtype, info=False):
    aterms = numpy.zeros(
        (nr_timeslots, nr_stations, subgrid_size, subgrid_size, nr_polarizations),
        dtype = atermtype)
    init_example_aterms(aterms, nr_timeslots, nr_stations, subgrid_size, subgrid_size)
    if info==True:
        print "aterms: numpy.ndarray(shape = (nr_timeslots, nr_stations," + \
              "subgrid_size, subgrid_size, nr_polarizations), " + \
              "dtype = " + str(dtype) + ")"
    return aterms.astype(dtype=dtype)


def get_example_aterms_offset(nr_timeslots, nr_time,
                              dtype=atermoffsettype, info=False):
    aterms_offset = numpy.zeros(
        (nr_timeslots + 1),
        dtype = atermoffsettype)
    init_example_aterms_offset(aterms_offset, nr_time)
    if info==True:
        print "aterms_offset: numpy.ndarray(shape = (nr_timeslots + 1), " + \
              "dtype = " + str(dtype) + ")"
    return aterms_offset.astype(dtype=dtype)


def get_example_spheroidal(subgrid_size,
                           dtype=spheroidaltype, info=False):
    spheroidal = numpy.ones((subgrid_size, subgrid_size),
                            dtype=spheroidaltype)
    init_example_spheroidal(spheroidal)
    if info==True:
        print "spheroidal: numpy.ndarray(shape = (subgrid_size, subgrid_size), " + \
              "dtype = " + str(dtype) + ")"
    return spheroidal.astype(dtype=dtype)


def get_example_visibilities(nr_baselines, nr_time, nr_channels,
                             nr_polarizations, image_size, grid_size,
                             uvw, frequencies,
                             nr_point_sources=4,
                             max_pixel_offset=-1,
                             random_seed=2,
                             dtype=visibilitiestype,
                             info=False):

    if max_pixel_offset==-1:
        max_pixel_offset = grid_size/2

    # Initialize visibilities to zero
    visibilities =  numpy.zeros(
        (nr_baselines, nr_time, nr_channels, nr_polarizations),
        dtype=visibilitiestype)

    # Create offsets for fake point sources
    offsets = list()
    random.seed(random_seed)
    for _ in range(nr_point_sources):
        x = (random.random() * (max_pixel_offset)) - (max_pixel_offset/2)
        y = (random.random() * (max_pixel_offset)) - (max_pixel_offset/2)
        offsets.append((x, y))

    # Update visibilities
    for offset in offsets:
        amplitude = 1
        add_pt_src(offset[0], offset[1], amplitude,
                   nr_baselines, nr_time, nr_channels, nr_polarizations,
                   image_size, grid_size, uvw, frequencies, visibilities)

    if info==True:
        print "spheroidal: numpy.ndarray(shape = (nr_baselines, nr_time, " + \
              "nr_channels, nr_polarizations), " + \
              "dtype = " + str(dtype) + ")"

    return visibilities.astype(dtype=dtype)

##### END: INITIALZE EXAMPLE DATA #####
