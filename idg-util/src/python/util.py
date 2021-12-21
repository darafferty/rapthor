# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

"""IDG utilities for generating examples"""

import math
import ctypes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import scipy.constants as sc
import numpy as np

from idg.idgtypes import *


# Load idg-util library
lib = ctypes.cdll.LoadLibrary('libidg-util.so')


def resize_spheroidal(spheroidal, size, dtype=np.float32):
    """
    Resize a spheroidal

    :param spheroidal: Input spheroidal
    :type spheroidal: np.arrayd(type=float) (two dimensions)
    :param size: New size along one axis
    :type size: int
    :param dtype: new dtype, defaults to np.float32
    :type dtype: np.dtype, optional
    :return: New spheroidal
    :rtype: np.array(dtype=float) (two dimensions)
    """
    if spheroidal.shape[1] != spheroidal.shape[0]:
        raise ValueError("Input spheroidal size should be square")
    subgrid_size = spheroidal.shape[0]
    tmp = spheroidal.astype(np.float32)
    result = np.zeros(shape=(size, size), dtype=np.float32)
    lib.utils_resize_spheroidal(tmp.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(subgrid_size),
                                result.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(size))
    return result.astype(dtype)


def nr_baselines_to_nr_stations(nr_baselines):
    """
    Convert number of baselines to number of stations, assuming that all station
    pairs are part of the baselines

    :param nr_baselines: Number of baselines
    :type nr_baselines: int
    :return: Number of stations
    :rtype: int
    """
    lower = int(math.floor(math.sqrt(2 * nr_baselines)))
    upper = int(math.ceil(math.sqrt(2 * nr_baselines) + 2))
    nr_stations = 2
    for i in range(lower, upper + 1):
        if (i * (i - 1) / 2 == nr_baselines):
            nr_stations = i
            return nr_stations
    return nr_stations


def add_pt_src(x, y, amplitude, nr_baselines, nr_time, nr_channels,
               nr_correlations, image_size, grid_size, uvw, frequencies, vis):
    """
    Add a point source to the model

    :param x: Source x location
    :type x: float
    :param y: Source y location
    :type y: float
    :param amplitude: [description]
    :type amplitude: float
    :param nr_baselines: [description]
    :type nr_baselines: int
    :param nr_time: [description]
    :type nr_time: int
    :param nr_channels: [description]
    :type nr_channels: int
    :param nr_correlations: [description]
    :type nr_correlations: int
    :param image_size: [description]
    :type image_size: float
    :param grid_size: [description]
    :type grid_size: int
    :param uvw: [description]
    :type uvw: np.array(dtype=float)
    :param frequencies: [description]
    :type frequencies: np.array(dtype=float)
    :param vis: [description]
    :type vis: np.array(dtype=complex)
    """

    lib.utils_add_pt_src.argtypes = [
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]

    lib.utils_add_pt_src(ctypes.c_float(x), ctypes.c_float(y),
                         ctypes.c_float(amplitude), ctypes.c_int(nr_baselines),
                         ctypes.c_int(nr_time), ctypes.c_int(nr_channels),
                         ctypes.c_int(nr_correlations),
                         ctypes.c_float(image_size), ctypes.c_int(grid_size),
                         uvw.ctypes.data_as(ctypes.c_void_p),
                         frequencies.ctypes.data_as(ctypes.c_void_p),
                         vis.ctypes.data_as(ctypes.c_void_p))


def func_spheroidal(nu):
    """Function to compute spheroidal
        Based on reference code by Bas"""
    P = np.array(
        [[8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
         [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2]])
    Q = np.array([[1.0000000e0, 8.212018e-1, 2.078043e-1],
                     [1.0000000e0, 9.599102e-1, 2.918724e-1]])

    part = 0
    end = 0.0

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
    for k in range(1, 5):
        top += P[part][k] * delnusqPow
        delnusqPow *= delnusq

    bot = Q[part][0]
    delnusqPow = delnusq
    for k in range(1, 3):
        bot += Q[part][k] * delnusqPow
        delnusqPow *= delnusq

    if bot == 0:
        result = 0
    else:
        result = (1.0 - nusq) * (top / bot)
    return result


def make_gaussian(size, fwhm=3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0)**2 + (y - y0)**2) / fwhm**2)


def init_example_spheroidal_subgrid(subgrid_size):
    """Construct spheroidal for subgrid"""
    # Spheroidal from Bas
    x = np.array(np.abs(
        np.linspace(-1, 1, num=subgrid_size, endpoint=False)),
                    dtype=np.float32)
    x = np.array([func_spheroidal(e) for e in x], dtype=np.float32)
    spheroidal = x[np.newaxis, :] * x[:, np.newaxis]
    return spheroidal
    # Ones
    #return np.ones((subgrid_size, subgrid_size), dtype = np.float32)
    # Gaussian
    #return make_gaussian(subgrid_size, int(subgrid_size * 0.3))


def init_example_spheroidal_grid(subgrid_size, grid_size):
    """Construct spheroidal for grid"""
    spheroidal = init_example_spheroidal_subgrid(subgrid_size)
    s = np.fft.fft2(spheroidal)
    s = np.fft.fftshift(s)
    s1 = np.zeros((grid_size, grid_size), dtype=np.complex64)
    support_size1 = int((grid_size - subgrid_size) / 2)
    support_size2 = int((grid_size + subgrid_size) / 2)
    s1[support_size1:support_size2, support_size1:support_size2] = s
    s1 = np.fft.ifftshift(s1)
    return np.real(np.fft.ifft2(s1))


def init_grid_of_point_sources(N,
                               image_size,
                               visibilities,
                               uvw,
                               frequencies,
                               asymmetric=False):
    """Initialize visibilities (and set w=0) to
    get a grid of N by N point sources

    Arguments:
    N - odd integer for N by N point sources
    image_size - ...
    visibilities - np.ndarray(shape=(nr_baselines, nr_time,
                                 nr_channels, nr_correlations),
                                 dtype=idg.visibilitiestype)
    uvw - np.ndarray(shape=(nr_baselines,nr_time),
                        dtype = idg.uvwtype)
    frequencies - np.ndarray(nr_channels, dtype = idg.frequenciestype)
    asymmetric - bool to make positive (l,m) twice in magnitude
    """

    # make sure N is odd, w=0, visibilities are zero initially
    if math.fmod(N, 2) == 0:
        N += 1
    uvw['w'] = 0
    visibilities.fill(0)

    # create visibilities
    nr_baselines = visibilities.shape[0]
    nr_time = visibilities.shape[1]
    nr_channels = visibilities.shape[2]
    nr_correlations = visibilities.shape[3]

    for b in range(nr_baselines):
        for t in range(nr_time):
            for c in range(nr_channels):
                u = frequencies[c] * uvw[b][t]['u'] / (sc.speed_of_light)
                v = frequencies[c] * uvw[b][t]['v'] / (sc.speed_of_light)
                for i in range(-N / 2 + 1,
                               N / 2 + 1):  # -N/2,-N/2+1,..,-1,0,1,...,N/2
                    for j in range(-N / 2 + 1,
                                   N / 2 + 1):  # -N/2,-N/2+1,..,-1,0,1,...,N/2
                        l = i * image_size / (N + 1)
                        m = j * image_size / (N + 1)
                        value = np.exp(
                            np.complex(0, -2 * np.pi * (u * l + v * m)))
                        if asymmetric == True:
                            if l > 0 and m > 0:
                                value *= 2
                        for p in range(nr_correlations):
                            visibilities[b][t][c][p] += value


##### BEGIN: PLOTTING UTILITY       #####


def get_figure_name(name):
    return "Figure %d: %s" % (len(plt.get_fignums()) + 1, name)


def plot_uvw_pixels(uvw, frequencies, image_size):
    """Plot UVW data as (u,v)-plot, scaled to pixel coordinates
    Input:
    uvw - np.ndarray(shape=(nr_subgrids, nr_timesteps, 3),
                        dtype = idg.uvwtype)
    """
    speed_of_light = 299792458.0
    u_ = np.array([], dtype=np.float32)
    v_ = np.array([], dtype=np.float32)
    for frequency in frequencies:
        u_ = np.append(
            u_, uvw['u'].flatten() * image_size * (frequency / speed_of_light))
        v_ = np.append(
            v_, uvw['v'].flatten() * image_size * (frequency / speed_of_light))
    uvlim = 1.2 * max(max(abs(u_)), max(abs(v_)))
    fig = plt.figure(get_figure_name("uvw"))
    plt.plot(u_, -v_, '.')
    plt.xlim([-uvlim, uvlim])
    plt.ylim([-uvlim, uvlim])
    plt.grid(True)
    fig.axes[0].set_aspect('equal')


def plot_tiles(uvw, frequencies, image_size, grid_size, tile_size=512):
    """Plot tiles corresponding to the UVW data, scaled to pixel coordinates
    Input:
    uvw - np.ndarray(shape=(nr_subgrids, nr_timesteps, 3),
                        dtype = idg.uvwtype)
    """

    # compute pixel coordinates
    speed_of_light = 299792458.0
    u_ = np.array([], dtype=np.float32)
    v_ = np.array([], dtype=np.float32)
    for frequency in frequencies:
        u_ = np.append(
            u_, uvw['u'].flatten() * image_size * (frequency / speed_of_light))
        v_ = np.append(
            v_, uvw['v'].flatten() * image_size * (frequency / speed_of_light))
    uvlim = max(max(abs(u_)), max(abs(v_)))
    assert (uvlim < grid_size)

    # determine which tiles are accessed
    nr_tiles_1d = int(grid_size / tile_size)
    tiles = np.zeros((nr_tiles_1d, nr_tiles_1d))
    for i in range(len(u_)):
        x = u_[i] + (grid_size / 2)
        y = v_[i] + (grid_size / 2)
        tiles[int(y / tile_size), int(x / tile_size)] += 1

    # compute percentage of tiles used
    nnz = np.sum(tiles > 0)
    percentage_used = np.round(nnz / (nr_tiles_1d**2) * 100, 2)
    title = "tiles: {}% used".format(percentage_used)

    # plot tiles
    tiles = np.log(tiles + 1)
    tiles[tiles == 0] = np.nan
    fig = plt.figure(get_figure_name(title))
    plt.imshow(tiles, interpolation='none')
    plt.colorbar()

    return percentage_used


def plot_tiles(metadata, image_size, grid_size, tile_size=512):
    """Plot tiles corresponding to the UVW data, scaled to pixel coordinates
    Input:
    uvw - np.ndarray(shape=(nr_subgrids, nr_timesteps, 3),
                        dtype = idg.uvwtype)
    """
    # get subgrid coordinates
    x = metadata['coordinate']['x'].flatten()
    y = metadata['coordinate']['y'].flatten()

    # determine which tiles are accessed
    nr_tiles_1d = int(grid_size / tile_size)
    tiles = np.zeros((nr_tiles_1d, nr_tiles_1d))
    for coordinate in zip(x, y):
        x = coordinate[0]
        y = coordinate[1]
        tiles[int(y / tile_size), int(x / tile_size)] += 1

    # compute percentage of tiles used
    nnz = np.sum(tiles > 0)
    percentage_used = np.round(nnz / (nr_tiles_1d**2) * 100, 2)
    title = "tiles: {}% used".format(percentage_used)

    # plot tiles
    tiles = np.log(tiles + 1)
    tiles[tiles == 0] = np.nan
    fig = plt.figure(get_figure_name(title))
    plt.imshow(tiles, interpolation='none')
    plt.colorbar()

    return percentage_used


def plot_uvw_meters(uvw):
    """Plot UVW data as (u,v)-plot
    Input:
    uvw - np.ndarray(shape=(nr_subgrids, nr_timesteps, 3),
                        dtype = idg.uvwtype)
    """
    u = uvw['u'].flatten()
    v = uvw['v'].flatten()
    uvlim = 1.2 * max(max(abs(u)), max(abs(v)))
    fig = plt.figure(get_figure_name("uvw"))
    plt.plot(u, -v, '.')
    plt.xlim([-uvlim, uvlim])
    plt.ylim([-uvlim, uvlim])
    plt.grid(True)
    fig.axes[0].set_aspect('equal')


def output_uvw(uvw):
    """Plot UVW data as (u,v)-plot to high-resolution png file
    Input:
    uvw - np.ndarray(shape=(nr_subgrids, nr_timesteps, 3),
                        dtype = idg.uvwtype)
    """
    u = uvw['u'].flatten()
    v = uvw['v'].flatten()
    fig = plt.figure(figsize=(40, 40), dpi=300)
    plt.plot(np.append(u, -u),
             np.append(v, -v),
             '.',
             color='black',
             alpha=0.8,
             markersize=1.0)
    plt.axes().set_aspect('equal')
    plt.axis(False)
    plt.savefig("uvw-coverage.png")


def plot_frequencies(frequencies):
    """Plot frequencies
    Input:
    frequencies - np.ndarray(nr_channels, dtype = idg.frequenciestype)
    """
    fig = plt.figure(get_figure_name("frequencies"))
    plt.plot(frequencies, '.')
    plt.grid(True)
    plt.xlabel("Channel")
    plt.ylabel("rad/m")


def plot_visibilities_all(visibilities, form='abs', maxtime=np.inf):
    """Plot visibility data
    Input:
    visibilities - np.ndarray(shape=(nr_baselines, nr_time,
                                nr_channels, nr_correlations),
                                dtype=idg.visibilitiestype)
    form - 'real', 'imag', 'abs', 'angle'
    """

    if maxtime > visibilities.shape[1]:
        maxtime = visibilities.shape[1] + 1

    if (form == 'real'):
        visXX = np.real(visibilities[:, :maxtime, :, 0].flatten())
        visXY = np.real(visibilities[:, :maxtime, :, 1].flatten())
        visYX = np.real(visibilities[:, :maxtime, :, 2].flatten())
        visYY = np.real(visibilities[:, :maxtime, :, 3].flatten())
        title = 'Real'
    elif (form == 'imag'):
        visXX = np.imag(visibilities[:, :maxtime, :, 0].flatten())
        visXY = np.imag(visibilities[:, :maxtime, :, 1].flatten())
        visYX = np.imag(visibilities[:, :maxtime, :, 2].flatten())
        visYY = np.imag(visibilities[:, :maxtime, :, 3].flatten())
        title = 'Imag'
    elif (form == 'angle'):
        visXX = np.angle(visibilities[:, :maxtime, :, 0].flatten())
        visXY = np.angle(visibilities[:, :maxtime, :, 1].flatten())
        visYX = np.angle(visibilities[:, :maxtime, :, 2].flatten())
        visYY = np.angle(visibilities[:, :maxtime, :, 3].flatten())
        title = 'Angle'
    else:
        visXX = np.abs(visibilities[:, :maxtime, :, 0].flatten())
        visXY = np.abs(visibilities[:, :maxtime, :, 1].flatten())
        visYX = np.abs(visibilities[:, :maxtime, :, 2].flatten())
        visYY = np.abs(visibilities[:, :maxtime, :, 3].flatten())
        title = 'Abs'

    fig, axarr = plt.subplots(2, 2, num=get_figure_name("visibilities"))
    fig.suptitle(title, fontsize=14)

    axarr[0, 0].plot(visXX)
    axarr[0, 1].plot(visXY)
    axarr[1, 0].plot(visYX)
    axarr[1, 1].plot(visYY)

    axarr[0, 0].set_title('XX')
    axarr[0, 1].set_title('XY')
    axarr[1, 0].set_title('YX')
    axarr[1, 1].set_title('YY')

    axarr[0, 0].tick_params(axis='both',
                            which='both',
                            bottom=False,
                            top=False,
                            labelbottom=False)

    axarr[0, 1].tick_params(axis='both',
                            which='both',
                            bottom=False,
                            top=False,
                            labelbottom=False)

    axarr[1, 0].tick_params(axis='both',
                            which='both',
                            bottom=False,
                            top=False,
                            labelbottom=False)

    axarr[1, 1].tick_params(axis='both',
                            which='both',
                            bottom=False,
                            top=False,
                            labelbottom=False)

def plot_visibilities(visibilities, form='abs', maxtime=np.inf):
    """Plot visibility data
    Input:
    visibilities - np.ndarray(shape=(nr_baselines, nr_time,
                                nr_channels, nr_correlations),
                                dtype=idg.visibilitiestype)
    form - 'real', 'imag', 'abs', 'angle'
    """
    nr_polarizations = visibilities.shape[3]
    if (nr_polarizations == 4):
        plot_visibilities_all(visibilities, form, maxtime)
        return
    else:
        pol = 0

    if maxtime > visibilities.shape[1]:
        maxtime = visibilities.shape[1] + 1

    if (form == 'real'):
        visXX = np.real(visibilities[:, :maxtime, :, 0].flatten())
        visYY = np.real(visibilities[:, :maxtime, :, 1].flatten())
        title = 'Real'
    elif (form == 'imag'):
        visXX = np.imag(visibilities[:, :maxtime, :, 0].flatten())
        visYY = np.imag(visibilities[:, :maxtime, :, 1].flatten())
        title = 'Imag'
    elif (form == 'angle'):
        visXX = np.angle(visibilities[:, :maxtime, :, 0].flatten())
        visYY = np.angle(visibilities[:, :maxtime, :, 1].flatten())
        title = 'Angle'
    else:
        visXX = np.abs(visibilities[:, :maxtime, :, 0].flatten())
        visYY = np.abs(visibilities[:, :maxtime, :, 1].flatten())
        title = 'Abs'

    fig, axarr = plt.subplots(2, num=get_figure_name("visibilities"))
    fig.suptitle(title, fontsize=14)

    axarr[0].plot(visXX)
    axarr[1].plot(visYY)

    axarr[0].set_title('XX')
    axarr[1].set_title('YY')

    axarr[0].tick_params(axis='both',
                         which='both',
                         bottom=False,
                         top=False,
                         labelbottom=False)

    axarr[1].tick_params(axis='both',
                         which='both',
                         bottom=False,
                         top=False,
                         labelbottom=False)

def plot_aterms(aterms):
    """Plot A-terms
    Input:
    aterms - np.ndarray(shape=(nr_timeslots, nr_stations,
                           subgrid_size, subgrid_size, nr_correlations),
                           dtype = idg.atermtype)
    """
    print("TO BE IMPLEMENTED")


def plot_spheroidal(spheroidal, interpolation_method='none'):
    """Plot spheroidal
    Input:
    spheroidal - np.ndarray(shape=(subgrid_size, subgrid_size),
                               dtype = idg.tapertype)
    interpolation_method - 'none', 'nearest', 'bilinear', 'bicubic',
                           'spline16', ... (see matplotlib imshow)
    """
    plt.figure(get_figure_name("spheroidal"))
    plt.imshow(spheroidal, interpolation=interpolation_method)
    plt.colorbar()


def plot_grid_all(grid,
                  form='abs',
                  scaling='none',
                  interpolation_method='none'):
    """Plot Grid data
    Input:
    grid - np.ndarray(shape=(nr_correlations, grid_size, grid_size),
                         dtype = idg.gridtype)
    form - 'real', 'imag', 'abs', 'angle'
    scaling - 'none', 'log', 'sqrt'
    interpolation_method - 'none', 'nearest', 'bilinear', 'bicubic',
                           'spline16', ... (see matplotlib imshow)
    """
    if (scaling == 'log'):
        grid = np.abs(grid) + 1
        grid = np.log(grid)
    if (scaling == 'sqrt'):
        grid = np.sqrt(grid)

    if (form == 'real'):
        gridXX = np.real(grid[0, :, :])
        gridXY = np.real(grid[1, :, :])
        gridYX = np.real(grid[2, :, :])
        gridYY = np.real(grid[3, :, :])
        title = 'Real'
    elif (form == 'imag'):
        gridXX = np.imag(grid[0, :, :])
        gridXY = np.imag(grid[1, :, :])
        gridYX = np.imag(grid[2, :, :])
        gridYY = np.imag(grid[3, :, :])
        title = 'Imag'
    elif (form == 'angle'):
        gridXX = np.angle(grid[0, :, :])
        gridXY = np.angle(grid[1, :, :])
        gridYX = np.angle(grid[2, :, :])
        gridYY = np.angle(grid[3, :, :])
        title = 'Angle'
    else:
        gridXX = np.abs(grid[0, :, :])
        gridXY = np.abs(grid[1, :, :])
        gridYX = np.abs(grid[2, :, :])
        gridYY = np.abs(grid[3, :, :])
        title = 'Abs'

    fig = plt.figure(get_figure_name("grid"))
    fig.suptitle(title, fontsize=14)

    ax = ["ax1", "ax2", "ax3", "ax4"]
    for idx in range(len(ax)):
        locals()[ax[idx]] = fig.add_subplot(2, 2, (idx + 1))
        divider = make_axes_locatable(vars()[ax[idx]])
        locals()["c" + ax[idx]] = divider.append_axes("right",
                                                      size="5%",
                                                      pad=0.05)

    im1 = locals()['ax1'].imshow(gridXX, interpolation=interpolation_method)
    plt.colorbar(im1, cax=locals()['cax1'], format='%.1e')

    im2 = locals()['ax2'].imshow(gridXY, interpolation=interpolation_method)
    plt.colorbar(im2, cax=locals()['cax2'], format='%.1e')

    im3 = locals()['ax3'].imshow(gridYX, interpolation=interpolation_method)
    plt.colorbar(im3, cax=locals()['cax3'], format='%.1e')

    im4 = locals()['ax4'].imshow(gridYY, interpolation=interpolation_method)
    plt.colorbar(im4, cax=locals()['cax4'], format='%.1e')

    locals()['ax1'].set_title('XX')
    locals()['ax2'].set_title('XY')
    locals()['ax3'].set_title('YX')
    locals()['ax4'].set_title('YY')

    locals()['ax1'].tick_params(axis='both',
                                which='both',
                                bottom=False,
                                top=False,
                                right=False,
                                left=False,
                                labelbottom=False,
                                labelleft=False)

    locals()['ax2'].tick_params(axis='both',
                                which='both',
                                bottom=False,
                                top=False,
                                right=False,
                                left=False,
                                labelbottom=False,
                                labelleft=False)

    locals()['ax3'].tick_params(axis='both',
                                which='both',
                                bottom=False,
                                top=False,
                                right=False,
                                left=False,
                                labelbottom=False,
                                labelleft=False)

    locals()['ax4'].tick_params(axis='both',
                                which='both',
                                bottom=False,
                                top=False,
                                right=False,
                                left=False,
                                labelbottom=False,
                                labelleft=False)


def plot_grid(grid,
              form='abs',
              scaling='none',
              interpolation_method='none',
              pol='all'):
    """Plot Grid data
    Input:
    grid - np.ndarray(shape=(nr_correlations, grid_size, grid_size),
                         dtype = idg.gridtype)
    form - 'real', 'imag', 'abs', 'angle'
    scaling - 'none', 'log', 'sqrt'
    interpolation_method - 'none', 'nearest', 'bilinear', 'bicubic',
                           'spline16', ... (see matplotlib imshow)
    """
    nr_polarizations = grid.shape[0]
    if (nr_polarizations == 4 and pol == 'all'):
        plot_grid_all(grid, form, scaling, interpolation_method)
        return
    else:
        pol = 0

    if (scaling == 'log'):
        grid = np.abs(grid) + 1
        grid = np.log(grid)
    if (scaling == 'sqrt'):
        grid = np.sqrt(grid)

    if (form == 'real'):
        grid = np.real(grid[pol, :, :])
        title = 'Real'
    elif (form == 'imag'):
        grid = np.imag(grid[pol, :, :])
        title = 'Imag'
    elif (form == 'angle'):
        grid = np.angle(grid[pol, :, :])
        title = 'Angle'
    else:
        grid = np.abs(grid[pol, :, :])
        title = 'Abs'

    fig = plt.figure(get_figure_name("grid"))
    fig.suptitle(title, fontsize=14)

    plt.imshow(grid, interpolation=interpolation_method)
    plt.colorbar(format='%.1e')

    plt.title(["XX", "XY", "YX", "YY"][pol])

    plt.tick_params(axis='both',
                    which='both',
                    bottom=False,
                    top=False,
                    right=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False)


def plot_metadata(metadata, uvw, frequencies, grid_size, subgrid_size,
                  image_size):
    # Show subgrids (from metadata)
    x = metadata['coordinate']['x'].flatten()
    y = metadata['coordinate']['y'].flatten()
    fig = plt.figure(get_figure_name("metadata: {} subgrids".format(len(x))))
    grid = np.zeros((grid_size, grid_size))
    for coordinate in zip(x, y):
        _x = coordinate[0]
        _y = coordinate[1]
        grid[_y:_y + subgrid_size, _x:_x + subgrid_size] += 1
    grid[grid == 0] = np.nan
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
    u_pixels = np.asarray(u_pixels).flatten() + (grid_size / 2)
    v_pixels = np.asarray(v_pixels).flatten() + (grid_size / 2)

    #plt.plot(u_pixels, v_pixels, 'r.', markersize=2, alpha=0.9)

    # Make mouseover show value of grid
    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if x > 0 and x < grid_size and \
           y > 0 and y < grid_size:
            z = grid[row, col]
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
    nr_timeslots = aterms.shape[0]
    nr_stations = aterms.shape[1]
    subgrid_size = aterms.shape[2]
    nr_correlations = aterms.shape[4]
    lib.utils_init_identity_aterms.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.utils_init_identity_aterms(aterms.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_int(nr_timeslots),
                                   ctypes.c_int(nr_stations),
                                   ctypes.c_int(subgrid_size),
                                   ctypes.c_int(nr_correlations))


def init_identity_spheroidal(spheroidal):
    subgrid_size = spheroidal.shape[0]
    lib.utils_init_identity_spheroidal.argtypes = [
        ctypes.c_void_p, ctypes.c_int
    ]
    lib.utils_init_identity_spheroidal(
        spheroidal.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(subgrid_size))


def get_identity_aterms(nr_timeslots,
                        nr_stations,
                        subgrid_size,
                        nr_correlations,
                        dtype=atermtype,
                        info=False):
    aterms = np.zeros((nr_timeslots, nr_stations, subgrid_size,
                          subgrid_size, nr_correlations),
                         dtype=atermtype)
    init_identity_aterms(aterms)
    if info == True:
        print("aterms: np.ndarray(shape = (nr_timeslots, nr_stations," + \
              "subgrid_size, subgrid_size, nr_correlations), " + \
              "dtype = " + str(dtype) + ")")
    return aterms.astype(dtype=dtype)


def get_zero_grid(nr_correlations, grid_size, dtype=gridtype, info=False):
    grid = np.zeros((nr_correlations, grid_size, grid_size), dtype=dtype)
    if info == True:
        print("grid: np.ndarray(shape = (nr_correlations, grid_size, grid_size), " + \
                                   "dtype = " + str(dtype) + ")")
    return grid


def get_identity_spheroidal(subgrid_size, dtype=tapertype, info=False):
    spheroidal = np.zeros(shape=(subgrid_size, subgrid_size),
                             dtype=tapertype)
    init_identity_spheroidal(spheroidal)
    if info == True:
        print("grid: np.ndarray(shape = (subgrid_size, subgrid_size), " + \
                                   "dtype = " + str(dtype) + ")")
    return spheroidal.astype(dtype=dtype)


def get_zero_visibilities(nr_baselines,
                          nr_time,
                          nr_channels,
                          nr_correlations,
                          dtype=visibilitiestype,
                          info=False):
    visibilities = np.zeros(shape=(nr_baselines, nr_time, nr_channels,
                                      nr_correlations),
                               dtype=visibilitiestype)
    return visibilities.astype(dtype=dtype)


##### END:   INITIALZE DATA         #####

##### BEGIN: INITIALZE EXAMPLE DATA #####


def init_example_frequencies(frequencies):
    """Initialize frequencies for test case defined in utility/initialize"""
    nr_channels = frequencies.shape[0]
    lib.utils_init_example_frequencies.argtypes = [
        ctypes.c_void_p, ctypes.c_int
    ]
    lib.utils_init_example_frequencies(
        frequencies.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nr_channels))


def init_dummy_visibilities(visibilities):
    """Initialize visibilities for test case defined in utility/initialize"""
    nr_baselines = visibilities.shape[0]
    nr_time = visibilities.shape[1]
    nr_channels = visibilities.shape[2]
    nr_correlations = visibilities.shape[3]
    lib.utils_init_example_visibilities.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.utils_init_dummy_visibilities(
        visibilities.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nr_baselines), ctypes.c_int(nr_time),
        ctypes.c_int(nr_channels), ctypes.c_int(nr_correlations))


def init_identity_aterms(aterms):
    """Initialize aterms for test case defined in utility/initialize"""
    nr_timeslots = aterms.shape[0]
    nr_stations = aterms.shape[1]
    subgrid_size = aterms.shape[2]
    nr_correlations = aterms.shape[4]
    lib.utils_init_identity_aterms.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.utils_init_identity_aterms(aterms.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_int(nr_timeslots),
                                   ctypes.c_int(nr_stations),
                                   ctypes.c_int(subgrid_size),
                                   ctypes.c_int(nr_correlations))


def init_example_spheroidal(spheroidal):
    """Initialize spheroidal for test case defined in utility/initialize"""
    subgrid_size = spheroidal.shape[0]
    lib.utils_init_example_spheroidal.argtypes = [
        ctypes.c_void_p, ctypes.c_int
    ]
    lib.utils_init_example_spheroidal(
        spheroidal.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(subgrid_size))


def init_example_aterms(aterms, nr_timeslots, nr_stations, height, width):
    """Initialize aterms"""
    lib.utils_init_example_aterms_offset.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.utils_init_example_aterms(aterms.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(nr_timeslots),
                                  ctypes.c_int(nr_stations),
                                  ctypes.c_int(height), ctypes.c_int(width))


def init_example_aterms_offset(aterms_offset, nr_time):
    """Initialize aterms offset"""
    nr_timeslots = aterms_offset.shape[0] - 1
    lib.utils_init_example_aterms_offset.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int
    ]
    lib.utils_init_example_aterms_offset(
        aterms_offset.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nr_timeslots), ctypes.c_int(nr_time))


def init_example_baselines(baselines, nr_stations):
    """Initialize baselines
    Input:
    baselines - np.ndarray(shape=(nr_baselines), dtype = idg.baselinetype)
    """
    nr_baselines = baselines.shape[0]
    lib.utils_init_example_baselines.argtypes = [ctypes.c_void_p,
                                                 ctypes.c_int,
                                                 ctypes.c_int]
    lib.utils_init_example_baselines(baselines.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_int(nr_stations),
                                     ctypes.c_int(nr_baselines))


def get_example_frequencies(nr_channels, dtype=frequenciestype, info=False):
    """Initialize and returns example frequencies array"""
    frequencies = np.ones(nr_channels, dtype=frequenciestype)
    init_example_frequencies(frequencies)
    if info == True:
        print("frequencies: np.ndarray(shape = (nr_channels), " + \
                                          "dtype = " + str(dtype) + ")")
    return frequencies.astype(dtype=dtype)


def get_example_baselines(nr_stations, nr_baselines,
                          dtype=np.int32, info=False):
    """Initialize and return example baselines array"""
    baselines = np.zeros((nr_baselines, 2),
                            dtype = np.int32)
    init_example_baselines(baselines, nr_stations)
    if info==True:
        print("baselines: np.ndarray(shape = (nr_channels), " + \
                                        "dtype = " + str(dtype) + ")")
    return baselines.astype(dtype=dtype)


def get_example_grid(nr_correlations, grid_size, dtype=gridtype, info=False):
    grid = np.zeros((nr_correlations, grid_size, grid_size), dtype=dtype)
    if info == True:
        print("grid: np.ndarray(shape = (nr_correlations, grid_size, grid_size), " + \
                                   "dtype = " + str(dtype) + ")")
    return grid


def get_example_aterms(nr_timeslots,
                       nr_stations,
                       subgrid_size,
                       nr_correlations,
                       dtype=atermtype,
                       info=False):
    aterms = np.zeros((nr_timeslots, nr_stations, subgrid_size,
                          subgrid_size, nr_correlations),
                         dtype=atermtype)
    init_example_aterms(aterms, nr_timeslots, nr_stations, subgrid_size,
                        subgrid_size)
    if info == True:
        print("aterms: np.ndarray(shape = (nr_timeslots, nr_stations," + \
              "subgrid_size, subgrid_size, nr_correlations), " + \
              "dtype = " + str(dtype) + ")")
    return aterms.astype(dtype=dtype)


def get_example_aterms_offset(nr_timeslots,
                              nr_time,
                              dtype=atermoffsettype,
                              info=False):
    aterms_offset = np.zeros((nr_timeslots + 1), dtype=atermoffsettype)
    init_example_aterms_offset(aterms_offset, nr_time)
    if info == True:
        print("aterms_offset: np.ndarray(shape = (nr_timeslots + 1), " + \
              "dtype = " + str(dtype) + ")")
    return aterms_offset.astype(dtype=dtype)


def get_example_spheroidal(subgrid_size, dtype=tapertype, info=False):
    spheroidal = np.ones((subgrid_size, subgrid_size), dtype=tapertype)
    init_example_spheroidal(spheroidal)
    if info == True:
        print("spheroidal: np.ndarray(shape = (subgrid_size, subgrid_size), " + \
              "dtype = " + str(dtype) + ")")
    return spheroidal.astype(dtype=dtype)


def get_example_visibilities(nr_baselines,
                             nr_time,
                             nr_channels,
                             nr_correlations,
                             image_size,
                             grid_size,
                             uvw,
                             frequencies,
                             nr_point_sources=4,
                             max_pixel_offset=-1,
                             random_seed=2,
                             dtype=visibilitiestype,
                             info=False):

    if max_pixel_offset == -1:
        max_pixel_offset = grid_size / 2

    # Initialize visibilities to zero
    visibilities = np.zeros(
        (nr_baselines, nr_time, nr_channels, nr_correlations),
        dtype=visibilitiestype)

    # Create offsets for fake point sources
    offsets = list()
    random.seed(random_seed)
    for _ in range(nr_point_sources):
        x = (random.random() * (max_pixel_offset)) - (max_pixel_offset / 2)
        y = (random.random() * (max_pixel_offset)) - (max_pixel_offset / 2)
        offsets.append((x, y))

    # Update visibilities
    for offset in offsets:
        amplitude = 1
        add_pt_src(offset[0], offset[1], amplitude, nr_baselines, nr_time,
                   nr_channels, nr_correlations, image_size, grid_size, uvw,
                   frequencies, visibilities)

    if info == True:
        print("spheroidal: np.ndarray(shape = (nr_baselines, nr_time, " + \
              "nr_channels, nr_correlations), " + \
              "dtype = " + str(dtype) + ")")

    return visibilities.astype(dtype=dtype)
