# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import ctypes
import numpy
import numpy.ctypeslib
import idg

lib = idg.load_library('libidg-common.so')

class Plan(object):

    def __init__(
        self,
        kernel_size,
        subgrid_size,
        grid_size,
        cell_size,
        frequencies,
        uvw,
        baselines,
        aterms_offsets):

        # extract dimensions
        nr_channels                  = frequencies.shape[0]
        uvw_nr_baselines             = uvw.shape[0]
        uvw_nr_timesteps             = uvw.shape[1]
        uvw_nr_coordinates           = 3
        baselines_nr_baselines       = baselines.shape[0]
        baselines_two                = 2
        aterms_offsets_nr_timeslots  = aterms_offsets.shape[0]

        lib.Plan_init.restype = ctypes.c_void_p
        lib.Plan_init.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_uint]
        self.obj = lib.Plan_init(
            ctypes.c_int(kernel_size),
            ctypes.c_int(subgrid_size),
            ctypes.c_int(grid_size),
            ctypes.c_float(cell_size),
            frequencies.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint(nr_channels),
            uvw.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint(uvw_nr_baselines),
            ctypes.c_uint(uvw_nr_timesteps),
            ctypes.c_uint(uvw_nr_coordinates),
            baselines.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint(baselines_nr_baselines),
            ctypes.c_uint(baselines_two),
            aterms_offsets.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint(aterms_offsets_nr_timeslots))


    def __del__(self):
        lib.Plan_destroy.argtypes = [ ctypes.c_void_p ]
        lib.Plan_destroy(self.obj)

    def get_nr_subgrids(self):
        lib.Plan_get_nr_subgrids.restype = ctypes.c_int
        lib.Plan_get_nr_subgrids.argtypes = [ ctypes.c_void_p ]
        return lib.Plan_get_nr_subgrids(self.obj)

    def copy_metadata(
        self,
        metadata):
        lib.Plan_copy_metadata.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p]
        lib.Plan_copy_metadata(
            self.obj,
            metadata.ctypes.data_as(ctypes.c_void_p))
