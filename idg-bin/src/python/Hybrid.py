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
# Currently only the hybrid-cuda library is supported
libpath = os.path.join(path, 'libidg-hybrid-cuda.so')
lib = ctypes.cdll.LoadLibrary(libpath)



class MaxwellHaswellEP(Proxy):
    """Hybrid Maxwell and HaswellEP implementation"""
    def __init__(self, nr_stations,
                       nr_channels,
                       nr_timesteps,
                       nr_timeslots,
                       imagesize,
                       grid_size,
                       subgrid_size = 32):
        try:
            lib.Hybrid_MaxwellHaswellEP_init.argtypes = [ctypes.c_uint, \
                                               ctypes.c_uint, \
                                               ctypes.c_uint, \
                                               ctypes.c_uint, \
                                               ctypes.c_float, \
                                               ctypes.c_uint, \
                                               ctypes.c_uint]
            self.obj = lib.Hybrid_MaxwellHaswellEP_init(
                ctypes.c_uint(nr_stations),
                ctypes.c_uint(nr_channels),
                ctypes.c_uint(nr_timesteps),
                ctypes.c_uint(nr_timeslots),
                ctypes.c_float(imagesize),
                ctypes.c_uint(grid_size),
                ctypes.c_uint(subgrid_size))
        except AttributeError:
            print "The chosen proxy was not built into the library"


    @classmethod
    def from_parameters(cls,p):
        """Another constructor form an instance of the Parameters class."""
        return cls(p.nr_stations, p.nr_channels, p.nr_timesteps, \
                   p.nr_timeslots, p.imagesize, p.grid_size, \
                   p.subgrid_size)

    def get_job_size_gridder(self):
        return lib.Hyrbrid_get_job_size_gridder(self.obj)

    def get_job_size_adder(self):
        return lib.Hyrbrid_get_job_size_adder(self.obj)

    def get_job_size_splitter(self):
        return lib.Hyrbrid_get_job_size_splitter(self.obj)

    def get_job_size_degridder(self):
        return lib.Hyrbrid_get_job_size_degridder(self.obj)

    def set_job_size_gridder(self, n = 8192):
        lib.Hyrbrid_set_job_size_gridder(self.obj, ctypes.c_int(n))

    def set_job_size_adder(self, n = 8192):
        lib.Hyrbrid_set_job_size_adder(self.obj, ctypes.c_int(n))

    def set_job_size_splitter(self, n = 8192):
        lib.Hyrbrid_set_job_size_splitter(self.obj, ctypes.c_int(n))

    def set_job_size_degridder(self, n = 8192):
        lib.Hyrbrid_set_job_size_degridder(self.obj, ctypes.c_int(n))

    # Wrapper to C function (override for each class inheriting from this)
    def _cwrap_grid_visibilities(self, visibilities, uvw, wavenumbers,
                                baselines, grid, w_offset, aterms,
                                spheroidal):
        lib.Hybrid_MaxwellHaswellEP_grid(self.obj,
                               visibilities.ctypes.data_as(ctypes.c_void_p),
                               uvw.ctypes.data_as(ctypes.c_void_p),
                               wavenumbers.ctypes.data_as(ctypes.c_void_p),
                               baselines.ctypes.data_as(ctypes.c_void_p),
                               grid.ctypes.data_as(ctypes.c_void_p),
                               ctypes.c_float(w_offset),
                               aterms.ctypes.data_as(ctypes.c_void_p),
                               spheroidal.ctypes.data_as(ctypes.c_void_p))

    # Wrapper to C function (override for each class inheriting from this)
    def _cwrap_degrid_visibilities(self, visibilities, uvw,
                                   wavenumbers, baselines,
                                   grid, w_offset, aterms,
                                   spheroidal):
        lib.Hybrid_MaxwellHaswellEP_degrid(self.obj,
                                 visibilities.ctypes.data_as(ctypes.c_void_p),
                                 uvw.ctypes.data_as(ctypes.c_void_p),
                                 wavenumbers.ctypes.data_as(ctypes.c_void_p),
                                 baselines.ctypes.data_as(ctypes.c_void_p),
                                 grid.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_float(w_offset),
                                 aterms.ctypes.data_as(ctypes.c_void_p),
                                 spheroidal.ctypes.data_as(ctypes.c_void_p))

    # Wrapper to C function (override for each class inheriting from this)
    def _cwrap_transform(self, direction, grid):
        lib.Hybrid_MaxwellHaswellEP_transform(self.obj,
                                    ctypes.c_int(direction),
                                    grid.ctypes.data_as(ctypes.c_void_p))


