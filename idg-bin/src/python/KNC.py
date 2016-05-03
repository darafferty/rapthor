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
libpath = os.path.join(path, 'libidg-knc.so')
lib = ctypes.cdll.LoadLibrary(libpath)



class Offload(Proxy):
    """KNC using offloading"""
    def __init__(self, nr_stations,
                       nr_channels,
                       nr_time,
                       nr_timeslots,
                       imagesize,
                       grid_size,
                       subgrid_size = 32):
        try:
            lib.KNC_Offload_init.argtypes = [ctypes.c_uint, \
                                               ctypes.c_uint, \
                                               ctypes.c_uint, \
                                               ctypes.c_uint, \
                                               ctypes.c_float, \
                                               ctypes.c_uint, \
                                               ctypes.c_uint]
            self.obj = lib.KNC_Offload_init(
                ctypes.c_uint(nr_stations),
                ctypes.c_uint(nr_channels),
                ctypes.c_uint(nr_time),
                ctypes.c_uint(nr_timeslots),
                ctypes.c_float(imagesize),
                ctypes.c_uint(grid_size),
                ctypes.c_uint(subgrid_size))
        except AttributeError:
            print "The chosen proxy was not built into the library"

    def _cwrap_grid_visibilities(self,
                                 visibilities,
                                 uvw,
                                 wavenumbers,
                                 baselines,
                                 grid,
                                 w_offset,
                                 kernel_size,
                                 aterms,
                                 aterms_offset,
                                 spheroidal):
            lib.KNC_Offload_grid(self.obj,
                               visibilities.ctypes.data_as(ctypes.c_void_p),
                               uvw.ctypes.data_as(ctypes.c_void_p),
                               wavenumbers.ctypes.data_as(ctypes.c_void_p),
                               baselines.ctypes.data_as(ctypes.c_void_p),
                               grid.ctypes.data_as(ctypes.c_void_p),
                               ctypes.c_float(w_offset),
                               ctypes.c_int(kernel_size),
                               aterms.ctypes.data_as(ctypes.c_void_p),
                               aterms_offset.ctypes.data_as(ctypes.c_void_p),
                               spheroidal.ctypes.data_as(ctypes.c_void_p))

    def _cwrap_degrid_visibilities(self,
                                   visibilities,
                                   uvw,
                                   wavenumbers,
                                   baselines,
                                   grid,
                                   w_offset,
                                   kernel_size,
                                   aterms,
                                   aterms_offset,
                                   spheroidal):
        lib.KNC_Offload_degrid(self.obj,
                                 visibilities.ctypes.data_as(ctypes.c_void_p),
                                 uvw.ctypes.data_as(ctypes.c_void_p),
                                 wavenumbers.ctypes.data_as(ctypes.c_void_p),
                                 baselines.ctypes.data_as(ctypes.c_void_p),
                                 grid.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_float(w_offset),
                                 ctypes.c_int(kernel_size),
                                 aterms.ctypes.data_as(ctypes.c_void_p),
                                 aterms_offset.ctypes.data_as(ctypes.c_void_p),
                                 spheroidal.ctypes.data_as(ctypes.c_void_p))


    def _cwrap_transform(self, direction, grid):
        lib.KNC_Offload_transform(self.obj,
                                    ctypes.c_int(direction),
                                    grid.ctypes.data_as(ctypes.c_void_p))
