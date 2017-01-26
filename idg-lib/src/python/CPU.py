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
libpath = os.path.join(path, 'libidg-cpu.so')
lib = ctypes.cdll.LoadLibrary(libpath)


class Reference(Proxy):

    def __init__(
        self,
        nr_correlations,
        subgrid_size):
        """Reference CPU implementation"""
        try:
            lib.CPU_Reference_init.argtypes = [ctypes.c_uint, \
                                               ctypes.c_uint]
            self.obj = lib.CPU_Reference_init(
                ctypes.c_uint(nr_correlations),
                ctypes.c_uint(subgrid_size))
        except AttributeError:
            print "The chosen proxy was not built into the library"


    def __del__(self):
        """Destroy"""
        lib.CPU_Reference_destroy(self.obj)


    # Wrapper to C function (override for each class inheriting from this)
    def _cwrap_grid_visibilities(
        self,
        w_offset,
        cell_size,
        kernel_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        grid,
        aterms,
        aterms_offset,
        spheroidal):
        lib.CPU_Reference_gridding(self.obj,
            ctypes.c_float(w_offset),
            ctypes.c_float(cell_size),
            ctypes.c_int(kernel_size),
            frequencies.ctypes.data_as(ctypes.c_void_p),
            visibilities.ctypes.data_as(ctypes.c_void_p),
            uvw.ctypes.data_as(ctypes.c_void_p),
            baselines.ctypes.data_as(ctypes.c_void_p),
            grid.ctypes.data_as(ctypes.c_void_p),
            aterms.ctypes.data_as(ctypes.c_void_p),
            aterms_offset.ctypes.data_as(ctypes.c_void_p),
            spheroidal.ctypes.data_as(ctypes.c_void_p))


    # Wrapper to C function (override for each class inheriting from this)
    def _cwrap_degridding(self,
        self,
        w_offset,
        cell_size,
        kernel_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        grid,
        aterms,
        aterms_offset,
        spheroidal):
        lib.CPU_Reference_gridding(self.obj,
            ctypes.c_float(w_offset),
            ctypes.c_float(cell_size),
            ctypes.c_int(kernel_size),
            frequencies.ctypes.data_as(ctypes.c_void_p),
            visibilities.ctypes.data_as(ctypes.c_void_p),
            uvw.ctypes.data_as(ctypes.c_void_p),
            baselines.ctypes.data_as(ctypes.c_void_p),
            grid.ctypes.data_as(ctypes.c_void_p),
            aterms.ctypes.data_as(ctypes.c_void_p),
            aterms_offset.ctypes.data_as(ctypes.c_void_p),
            spheroidal.ctypes.data_as(ctypes.c_void_p))


    # Wrapper to C function (override for each class inheriting from this)
    def _cwrap_transform(self, direction, grid):
        lib.CPU_Reference_transform(
            self.obj,
            ctypes.c_int(direction),
            grid.ctypes.data_as(ctypes.c_void_p))
