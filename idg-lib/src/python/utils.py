import os
import math
import numpy
import ctypes
import numpy.ctypeslib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# A bit ugly, but ctypes.util's find_library does not look in
# the LD_LIBRARY_PATH, but only PATH. Howver, we can also provide
# the full path of the shared object file
path = os.path.dirname(os.path.realpath(__file__))
path, junk = os.path.split(path)
path, junk = os.path.split(path)
libpath = os.path.join(path, 'libidg-utility.so')
lib = ctypes.cdll.LoadLibrary(libpath)

def get_figure_name(name):
    return "Figure %d: %s" % (len(plt.get_fignums()) + 1, name)

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


def init_uvw(uvw, integration_time = 10):
    """Initialize uvw for test case defined in utility/initialize"""
    nr_baselines = uvw.shape[0]
    nr_stations = nr_baselines_to_nr_stations(nr_baselines)
    nr_time = uvw.shape[1]
    lib.utils_init_uvw.argtypes = [ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int]
    lib.utils_init_uvw( uvw.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nr_stations),
                        ctypes.c_int(nr_baselines),
                        ctypes.c_int(nr_time),
                        ctypes.c_int(integration_time))


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


def init_wavenumbers(wavenumbers):
    """Initialize wavenumbers for test case defined in utility/initialize"""
    nr_channels = wavenumbers.shape[0]
    lib.utils_init_wavenumbers.argtypes = [ctypes.c_void_p,
                                           ctypes.c_int]
    lib.utils_init_wavenumbers(wavenumbers.ctypes.data_as(ctypes.c_void_p),
                               ctypes.c_int(nr_channels) )


def plot_wavenumbers(wavenumbers):
    """Plot wavenumbers
    Input:
    wavenumbers - numpy.ndarray(nr_channels, dtype = idg.wavenumberstype)
    """
    fig = plt.figure(get_figure_name("wavenumbers"))
    plt.plot(wavenumbers,'.')
    plt.grid(True)
    plt.xlabel("Channel")
    plt.ylabel("rad/m")


def init_metadata(metadata, uvw, wavenumbers, nr_timesteps,
                  nr_timeslots, image_size, grid_size, subgrid_size):
    """Initialize wavenumbers for test case defined in utility/initialize"""
    nr_baselines = uvw.shape[0]
    nr_stations = nr_baselines_to_nr_stations(nr_baselines)
    nr_channels = wavenumbers.shape[0]

    lib.utils_init_metadata.argtypes = [ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_float]
    lib.utils_init_metadata( metadata.ctypes.data_as(ctypes.c_void_p),
                             uvw.ctypes.data_as(ctypes.c_void_p),
                             wavenumbers.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(nr_stations),
                             ctypes.c_int(nr_baselines),
                             ctypes.c_int(nr_timesteps),
                             ctypes.c_int(nr_timeslots),
                             ctypes.c_int(nr_channels),
                             ctypes.c_int(grid_size),
                             ctypes.c_int(subgrid_size),
                             ctypes.c_float(image_size) )


def init_visibilities(visibilities):
    """Initialize visibilities for test case defined in utility/initialize"""
    nr_baselines = visibilities.shape[0]
    nr_time = visibilities.shape[1]
    nr_channels = visibilities.shape[2]
    nr_polarizations = visibilities.shape[3]
    lib.utils_init_visibilities.argtypes = [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int]
    lib.utils_init_visibilities(visibilities.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(nr_baselines),
                                ctypes.c_int(nr_time),
                                ctypes.c_int(nr_channels),
                                ctypes.c_int(nr_polarizations) )


def plot_visibilities(visibilities, form='abs'):
    """Plot Grid data
    Input:
    visibilities - numpy.ndarray(shape=(nr_baselines, nr_time,
                                nr_channels, nr_polarizations),
                                dtype=idg.visibilitiestype)
    form - 'real', 'imag', 'abs', 'phase'
    """
    if (form=='real'):
        visXX = numpy.real( visibilities[:,:,:,0].flatten() )
        visXY = numpy.real( visibilities[:,:,:,1].flatten() )
        visYX = numpy.real( visibilities[:,:,:,2].flatten() )
        visYY = numpy.real( visibilities[:,:,:,3].flatten() )
        title = 'Real'
    if (form=='imag'):
        visXX = numpy.imag( visibilities[:,:,:,0].flatten() )
        visXY = numpy.imag( visibilities[:,:,:,1].flatten() )
        visYX = numpy.imag( visibilities[:,:,:,2].flatten() )
        visYY = numpy.imag( visibilities[:,:,:,3].flatten() )
        title = 'Imag'
    if (form=='angle'):
        visXX = numpy.angle( visibilities[:,:,:,0].flatten() )
        visXY = numpy.angle( visibilities[:,:,:,1].flatten() )
        visYX = numpy.angle( visibilities[:,:,:,2].flatten() )
        visYY = numpy.angle( visibilities[:,:,:,3].flatten() )
        title = 'Angle'
    else:
        visXX = numpy.abs( visibilities[:,:,:,0].flatten() )
        visXY = numpy.abs( visibilities[:,:,:,1].flatten() )
        visYX = numpy.abs( visibilities[:,:,:,2].flatten() )
        visYY = numpy.abs( visibilities[:,:,:,3].flatten() )
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


def init_aterms(aterms):
    """Initialize aterms for test case defined in utility/initialize"""
    nr_stations = aterms.shape[0]
    nr_timeslots = aterms.shape[1]
    nr_polarizations = aterms.shape[2]
    subgrid_size = aterms.shape[3]
    lib.utils_init_aterms.argtypes = [ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int]
    lib.utils_init_aterms(aterms.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(nr_stations),
                          ctypes.c_int(nr_timeslots),
                          ctypes.c_int(nr_polarizations),
                          ctypes.c_int(subgrid_size) )

def plot_aterms(aterms):
    """Plot A-terms
    Input:
    aterms - numpy.ndarray(shape=(nr_stations, nr_timeslots,
                           nr_polarizations, subgrid_size, subgrid_size),
                           dtype = idg.atermtype)
    """
    print "TO BE IMPLEMENTED"


def init_spheroidal(spheroidal):
    """Initialize aterms for test case defined in utility/initialize"""
    subgrid_size = spheroidal.shape[0]
    lib.utils_init_spheroidal.argtypes = [ctypes.c_void_p,
                                          ctypes.c_int]
    lib.utils_init_spheroidal(spheroidal.ctypes.data_as(ctypes.c_void_p),
                              ctypes.c_int(subgrid_size) )


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

def init_baselines(baselines):
    """Initialize baselines
    Input:
    baselines - numpy.ndarray(shape=(nr_baselines), dtype = idg.baselinetype)
    """
    nr_baselines = baselines.shape[0]
    nr_stations = nr_baselines_to_nr_stations(nr_baselines)
    lib.utils_init_baselines.argtypes = [ctypes.c_void_p,
                                         ctypes.c_int,
                                         ctypes.c_int]
    lib.utils_init_baselines(baselines.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(nr_stations),
                             ctypes.c_int(nr_baselines))

def plot_grid(grid, form='abs', scaling='none', interpolation_method='none'):
    """Plot Grid data
    Input:
    grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
                         dtype = idg.gridtype)
    form - 'real', 'imag', 'abs', 'phase'
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


def plot_metadata(metadata, uvw, wavenumbers, grid_size, subgrid_size, image_size):
    # Show subgrids (from metadata)
    x = metadata['coordinate']['x'].flatten()
    y = metadata['coordinate']['y'].flatten()
    fig = plt.figure(get_figure_name("metadata"))
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
    for wavenumber in wavenumbers:
        scaling = (wavenumber * image_size / (2 * numpy.pi))
        u_pixels.append(u * scaling)
        v_pixels.append(v * scaling)
    u_pixels = numpy.asarray(u_pixels).flatten() + (grid_size / 2)
    v_pixels = numpy.asarray(v_pixels).flatten() + (grid_size / 2)
    plt.plot(u_pixels, v_pixels, 'r.', markersize=1, alpha=0.1)

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


def init_grid_of_point_sources(N, image_size, visibilities, uvw,
                               wavenumbers, asymmetric=False):
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
    wavenumbers - numpy.ndarray(nr_channels, dtype = idg.wavenumberstype)
    asymmetric - bool to make positive (l,m) twice in magnitude
    """

    # make sure N is odd, w=0, visibilities are zero initially
    if math.fmod(N,2)==0:
        N += 1
    uvw['w'] = 0
    visibilities.fill(0)

    # create visibilities
    nr_baselines = visibilities.shape[0]
    nr_time = visibilities.shape[1]
    nr_channels = visibilities.shape[2]
    nr_polarizations = visibilities.shape[3]

    for b in range(nr_baselines):
        for t in range(nr_time):
            for c in range(nr_channels):
                u = wavenumbers[c]*uvw[b][t]['u']/(2*numpy.pi)
                v = wavenumbers[c]*uvw[b][t]['v']/(2*numpy.pi)
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
