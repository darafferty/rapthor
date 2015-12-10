import os
import ctypes
import numpy.ctypeslib

# A bit ugly, but ctypes.util's find_library does not look in
# the LD_LIBRARY_PATH, but only PATH. Howver, we can also provide
# the full path of the shared object file
path = os.path.dirname(os.path.realpath(__file__))
path, junk = os.path.split(path)
path, junk = os.path.split(path)
libpath = os.path.join(path, 'libidg-utility.so')
lib = ctypes.cdll.LoadLibrary(libpath)


def init_uvw(uvw, nr_stations, nr_baselines, nr_time, gridsize,
            subgridsize):
    print uvw.shape

    nr_baselines = nr_stations * (nr_stations-1) / 2

    lib.utils_init_uvw = [ctypes.c_void_p, \
                          ctypes.c_int, \
                          ctypes.c_int, \
                          ctypes.c_int, \
                          ctypes.c_int, \
                          ctypes.c_int]
    lib.utils_init_uvw(uvw.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_uint(nr_stations),
                       ctypes.c_uint(nr_baselines),
                       ctypes.c_uint(nr_time),
                       ctypes.c_uint(gridsize),
                       ctypes.c_uint(subgridsize))
