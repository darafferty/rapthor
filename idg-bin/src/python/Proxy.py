import os
import ctypes

# A bit ugly, but ctypes.util's find_library does not look in
# the LD_LIBRARY_PATH, but only PATH. Howver, we can also provide
# the full path of the shared object file
path = os.path.dirname(os.path.realpath(__file__))
path, junk = os.path.split(path)
path, junk = os.path.split(path)
libpath = os.path.join(path, 'libidg.so')
lib = ctypes.cdll.LoadLibrary(libpath)

class Proxy(object):
    def get_nr_stations(self):
        return lib.Proxy_get_nr_stations(self.obj)

    def get_nr_baselines(self):
        return lib.Proxy_get_nr_baselines(self.obj)

    def get_nr_channels(self):
        return lib.Proxy_get_nr_channels(self.obj)

    def get_nr_timesteps(self):
        return lib.Proxy_get_nr_timesteps(self.obj)

    def get_nr_timeslots(self):
        return lib.Proxy_get_nr_timeslots(self.obj)

    def get_nr_polarizations(self):
        return lib.Proxy_get_nr_polarizations(self.obj)

    def get_imagesize(self):
        lib.Proxy_get_imagesize.restype = ctypes.c_float
        return lib.Proxy_get_imagesize(self.obj)

    def get_grid_size(self):
        return lib.Proxy_get_grid_size(self.obj)

    def get_subgrid_size(self):
        return lib.Proxy_get_subgrid_size(self.obj)

    def get_nr_subgrids(self):
        return lib.Proxy_get_nr_subgrids(self.obj)

    def get_job_size(self):
        return lib.Proxy_get_job_size(self.obj)

    def get_job_size_gridding(self):
        return lib.Proxy_get_job_size_gridding(self.obj)

    def get_job_size_degridding(self):
        return lib.Proxy_get_job_size_degridding(self.obj)

    def set_job_size(self, n = 8192):
        lib.Proxy_set_job_size(self.obj, ctypes.cint(n))

    def set_job_size_gridding(self, n = 8192):
        lib.Proxy_set_job_size_gridding(self.obj, ctypes.c_int(n))

    def set_job_size_degridding(self, n = 8192):
        lib.Proxy_set_job_size_degridding(self.obj, ctypes.c_int(n))
