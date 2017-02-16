import os
import ctypes
import numpy.ctypeslib
from Proxy import *


# A bit ugly, but ctypes.util's find_library does not look in
# the LD_LIBRARY_PATH, but only PATH. Howver, we can also provide
# the full path of the shared object file
path = os.path.dirname(os.path.realpath(__file__))
path, junk = os.path.split(path)
path, junk = os.path.split(path)
libpath = os.path.join(path, 'libidg-hybrid-cuda.so')
lib = ctypes.cdll.LoadLibrary(libpath)


class GenericOptimized(Proxy):

    def __init__(
        self,
        nr_correlations,
        subgrid_size):
        """GenericOptimized CUDA implementation"""
        try:
            lib.HybridCUDA_GenericOptimized_init.argtypes = [ctypes.c_uint, \
                                               ctypes.c_uint]
            self.obj = lib.HybridCUDA_GenericOptimized_init(
                ctypes.c_uint(nr_correlations),
                ctypes.c_uint(subgrid_size))
        except AttributeError:
            print "The chosen proxy was not built into the library"


    def __del__(self):
        """Destroy"""
        lib.HybridCUDA_GenericOptimized_destroy(self.obj)


    def _cwrap_griddding(
        self,
        w_offset,
        cell_size,
        kernel_size,
        frequencies,
        nr_channels,
        visibilities,
        visibilities_nr_baselines,
        visibilities_nr_timesteps,
        visibilities_nr_channels,
        visibilities_nr_correlations,
        uvw,
        uvw_nr_baselines,
        uvw_nr_timesteps,
        uvw_nr_coordinates,
        baselines,
        baselines_nr_baselines,
        baselines_two,
        grid,
        grid_nr_correlations,
        grid_height, grid_width, aterms, aterms_nr_timeslots,
        aterms_nr_stations,
        aterms_aterm_height,
        aterms_aterm_width,
        aterms_nr_correlations,
        aterms_offsets,
        aterms_offsets_nr_timeslots,
        spheroidal,
        spheroidal_height,
        spheroidal_width):
        lib.HybridCUDA_GenericOptimized_gridding(
            self.obj,
            ctypes.c_float(w_offset),
            ctypes.c_float(cell_size),
            ctypes.c_int(kernel_size),
            frequencies.ctypes.data_as(ctypes.c_void_p),
            nr_channels,
            visibilities.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(visibilities_nr_baselines),
            ctypes.c_int(visibilities_nr_timesteps),
            ctypes.c_int(visibilities_nr_channels),
            ctypes.c_int(visibilities_nr_correlations),
            uvw.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(uvw_nr_baselines),
            ctypes.c_int(uvw_nr_timesteps),
            ctypes.c_int(uvw_nr_coordinates),
            baselines.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(baselines_nr_baselines),
            ctypes.c_int(baselines_two),
            grid.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(grid_nr_correlations),
            ctypes.c_int(grid_height),
            ctypes.c_int(grid_width),
            aterms.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(aterms_nr_timeslots),
            ctypes.c_int(aterms_nr_stations),
            ctypes.c_int(aterms_aterm_height),
            ctypes.c_int(aterms_aterm_width),
            ctypes.c_int(aterms_nr_correlations),
            aterms_offsets.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(aterms_offsets_nr_timeslots),
            spheroidal.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(spheroidal_height),
            ctypes.c_int(spheroidal_width))


    def _cwrap_degridding(
        self,
        w_offset,
        cell_size,
        kernel_size,
        frequencies,
        nr_channels,
        visibilities,
        visibilities_nr_baselines,
        visibilities_nr_timesteps,
        visibilities_nr_channels,
        visibilities_nr_correlations,
        uvw,
        uvw_nr_baselines,
        uvw_nr_timesteps,
        uvw_nr_coordinates,
        baselines,
        baselines_nr_baselines,
        baselines_two,
        grid,
        grid_nr_correlations,
        grid_height,
        grid_width,
        aterms,
        aterms_nr_timeslots,
        aterms_nr_stations,
        aterms_aterm_height,
        aterms_aterm_width,
        aterms_nr_correlations,
        aterms_offsets,
        aterms_offsets_nr_timeslots,
        spheroidal,
        spheroidal_height,
        spheroidal_width):
        lib.HybridCUDA_GenericOptimized_degridding(
            self.obj,
            ctypes.c_float(w_offset),
            ctypes.c_float(cell_size),
            ctypes.c_int(kernel_size),
            frequencies.ctypes.data_as(ctypes.c_void_p),
            nr_channels,
            visibilities.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(visibilities_nr_baselines),
            ctypes.c_int(visibilities_nr_timesteps),
            ctypes.c_int(visibilities_nr_channels),
            ctypes.c_int(visibilities_nr_correlations),
            uvw.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(uvw_nr_baselines),
            ctypes.c_int(uvw_nr_timesteps),
            ctypes.c_int(uvw_nr_coordinates),
            baselines.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(baselines_nr_baselines),
            ctypes.c_int(baselines_two),
            grid.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(grid_nr_correlations),
            ctypes.c_int(grid_height),
            ctypes.c_int(grid_width),
            aterms.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(aterms_nr_timeslots),
            ctypes.c_int(aterms_nr_stations),
            ctypes.c_int(aterms_aterm_height),
            ctypes.c_int(aterms_aterm_width),
            ctypes.c_int(aterms_nr_correlations),
            aterms_offsets.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(aterms_offsets_nr_timeslots),
            spheroidal.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(spheroidal_height),
            ctypes.c_int(spheroidal_width))


    def _cwrap_transform(
        self,
        direction,
        grid,
        nr_correlations,
        height,
        width):
        lib.HybridCUDA_GenericOptimized_transform(
            self.obj,
            ctypes.c_int(direction),
            grid.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nr_correlations),
            ctypes.c_int(height),
            ctypes.c_int(width))
