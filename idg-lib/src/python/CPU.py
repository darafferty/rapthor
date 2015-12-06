import os
import ctypes
import numpy.ctypeslib

# A bit ugly, but ctypes.util's find_library does not look in
# the LD_LIBRARY_PATH, but only PATH. Howver, we can also provide
# the full path of the shared object file
path = os.path.dirname(os.path.realpath(__file__))
path, junk = os.path.split(path)
path, junk = os.path.split(path)
libpath = os.path.join(path, 'libidg.so')
lib = ctypes.cdll.LoadLibrary(libpath)


class Reference(object):
    def __init__(self, nr_stations,
                       nr_channels,
                       nr_timesteps,
                       nr_timeslots,
                       imagesize,
                       grid_size,
                       subgrid_size = 32):
        try:
            lib.CPU_Reference_init.argtypes = [ctypes.c_uint, \
                                               ctypes.c_uint, \
                                               ctypes.c_uint, \
                                               ctypes.c_uint, \
                                               ctypes.c_float, \
                                               ctypes.c_uint, \
                                               ctypes.c_uint]
            self.obj = lib.CPU_Reference_init(
                ctypes.c_uint(nr_stations),
                ctypes.c_uint(nr_channels),
                ctypes.c_uint(nr_timesteps),
                ctypes.c_uint(nr_timeslots),
                ctypes.c_float(imagesize),
                ctypes.c_uint(grid_size),
                ctypes.c_uint(subgrid_size))
        except AttributeError:
            print "The chosen proxy was not built into the library"

    def grid_visibilities(self,
                          visibilities,
                          uvw,
                          wavenumbers,
                          metadata,
                          grid,
                          w_offset,
                          aterm,
                          spheroidal):
        lib.CPU_Reference_grid(self.obj,
                               visibilities.ctypes.data_as(ctypes.c_void_p),
                               uvw.ctypes.data_as(ctypes.c_void_p),
                               wavenumbers.ctypes.data_as(ctypes.c_void_p),
                               metadata.ctypes.data_as(ctypes.c_void_p),
                               grid.ctypes.data_as(ctypes.c_void_p),
                               ctypes.c_float(w_offset),
                               aterm.ctypes.data_as(ctypes.c_void_p),
                               spheroidal.ctypes.data_as(ctypes.c_void_p))

    def degrid_visibilities(self,
                            visibilities,
                            uvw,
                            wavenumbers,
                            metadata,
                            grid,
                            w_offset,
                            aterm,
                            spheroidal):
        lib.CPU_Reference_degrid(self.obj,
                                 visibilities.ctypes.data_as(ctypes.c_void_p),
                                 uvw.ctypes.data_as(ctypes.c_void_p),
                                 wavenumbers.ctypes.data_as(ctypes.c_void_p),
                                 metadata.ctypes.data_as(ctypes.c_void_p),
                                 grid.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_float(w_offset),
                                 aterm.ctypes.data_as(ctypes.c_void_p),
                                 spheroidal.ctypes.data_as(ctypes.c_void_p))

    def transform(self,
                  direction,
                  grid):
        lib.CPU_Reference_transform(self.obj,
                                    ctypes.c_int(direction),
                                    grid.ctypes.data_as(ctypes.c_void_p))

    def get_nr_stations(self):
        return lib.CPU_Reference_get_nr_stations(self.obj)

    def get_nr_baselines(self):
        return lib.CPU_Reference_get_nr_baselines(self.obj)

    def get_nr_channels(self):
        return lib.CPU_Reference_get_nr_channels(self.obj)

    def get_nr_timesteps(self):
        return lib.CPU_Reference_get_nr_timesteps(self.obj)

    def get_nr_timeslots(self):
        return lib.CPU_Reference_get_nr_timeslots(self.obj)

    def get_nr_polarizations(self):
        return lib.CPU_Reference_get_nr_polarizations(self.obj)

    def get_imagesize(self):
        lib.CPU_Reference_get_imagesize.restype = ctypes.c_float
        return lib.CPU_Reference_get_imagesize(self.obj)

    def get_grid_size(self):
        return lib.CPU_Reference_get_grid_size(self.obj)

    def get_subgrid_size(self):
        return lib.CPU_Reference_get_subgrid_size(self.obj)

    def get_nr_subgrids(self):
        return lib.CPU_Reference_get_nr_subgrids(self.obj)

    def get_job_size(self):
        return lib.CPU_Reference_get_job_size(self.obj)

    def get_job_size_gridding(self):
        return lib.CPU_Reference_get_job_size_gridding(self.obj)

    def get_job_size_degridding(self):
        return lib.CPU_Reference_get_job_size_degridding(self.obj)

    def get_job_size_gridder(self):
        return lib.CPU_Reference_get_job_size_gridder(self.obj)

    def get_job_size_adder(self):
        return lib.CPU_Reference_get_job_size_adder(self.obj)

    def get_job_size_splitter(self):
        return lib.CPU_Reference_get_job_size_splitter(self.obj)

    def get_job_size_degridder(self):
        return lib.CPU_Reference_get_job_size_degridder(self.obj)

    def set_job_size(self, n = 8192):
        lib.CPU_Reference_set_job_size(self.obj, ctypes.cint(n))

    def set_job_size_gridding(self, n = 8192):
        lib.CPU_Reference_set_job_size_gridding(self.obj, ctypes.c_int(n))

    def set_job_size_degridding(self, n = 8192):
        lib.CPU_Reference_set_job_size_degridding(self.obj, ctypes.c_int(n))

    def set_job_size_gridder(self, n = 8192):
        lib.CPU_Reference_set_job_size_gridder(self.obj, ctypes.c_int(n))

    def set_job_size_adder(self, n = 8192):
        lib.CPU_Reference_set_job_size_adder(self.obj, ctypes.c_int(n))

    def set_job_size_splitter(self, n = 8192):
        lib.CPU_Reference_set_job_size_splitter(self.obj, ctypes.c_int(n))

    def set_job_size_degridder(self, n = 8192):
        lib.CPU_Reference_set_job_size_degridder(self.obj, ctypes.c_int(n))
