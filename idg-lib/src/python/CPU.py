import os
import ctypes
import numpy.ctypeslib
from Proxy import *

class CPU(Proxy):
    lib = idg.load_library('libidg-cpu.so')

class Reference(CPU):

    def __init__(
        self,
        nr_correlations,
        subgrid_size):
        """Reference CPU implementation"""
        print self.lib
        try:
            self.lib.CPU_Reference_init.argtypes = [ctypes.c_uint, \
                                               ctypes.c_uint]
            self.obj = self.lib.CPU_Reference_init(
                ctypes.c_uint(nr_correlations),
                ctypes.c_uint(subgrid_size))
        except AttributeError:
            print "The chosen proxy was not built into the library"


    def __del__(self):
        """Destroy"""
        self.lib.CPU_Reference_destroy(self.obj)


    def _cwrap_griddding(
        self,
        w_step,
        shift,
        cell_size,
        kernel_size,
        subgrid_size,
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
        self.lib.CPU_Reference_gridding(
            self.obj,
            ctypes.c_float(w_step),
            shift.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_float(cell_size),
            ctypes.c_int(kernel_size),
            ctypes.c_int(subgrid_size),
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
        w_step,
        shift,
        cell_size,
        kernel_size,
        subgrid_size,
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
        self.lib.CPU_Reference_degridding(
            self.obj,
            ctypes.c_float(w_step),
            shift.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_float(cell_size),
            ctypes.c_int(kernel_size),
            ctypes.c_int(subgrid_size),
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
        self.lib.CPU_Reference_transform(
            self.obj,
            ctypes.c_int(direction),
            grid.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nr_correlations),
            ctypes.c_int(height),
            ctypes.c_int(width))



class Optimized(CPU):

    def __init__(
        self,
        nr_correlations,
        subgrid_size):
        """Optimized CPU implementation"""
        try:
            self.lib.CPU_Optimized_init.argtypes = [ctypes.c_uint, \
                                               ctypes.c_uint]
            self.obj = self.lib.CPU_Optimized_init(
                ctypes.c_uint(nr_correlations),
                ctypes.c_uint(subgrid_size))
        except AttributeError:
            print "The chosen proxy was not built into the library"


    def __del__(self):
        """Destroy"""
        self.lib.CPU_Optimized_destroy(self.obj)


    def _cwrap_griddding(
        self,
        w_step,
        shift,
        cell_size,
        kernel_size,
        subgrid_size,
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
        self.lib.CPU_Optimized_gridding(
            self.obj,
            ctypes.c_float(w_step),
            shift.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_float(cell_size),
            ctypes.c_int(kernel_size),
            ctypes.c_int(subgrid_size),
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
        w_step,
        shift,
        cell_size,
        kernel_size,
        subgrid_size,
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
        self.lib.CPU_Optimized_degridding(
            self.obj,
            ctypes.c_float(w_step),
            shift.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_float(cell_size),
            ctypes.c_int(kernel_size),
            ctypes.c_int(subgrid_size),
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
        self.lib.CPU_Optimized_transform(
            self.obj,
            ctypes.c_int(direction),
            grid.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nr_correlations),
            ctypes.c_int(height),
            ctypes.c_int(width))
