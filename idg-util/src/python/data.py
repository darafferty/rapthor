import numpy
import ctypes
import numpy.ctypeslib

from idgtypes import *

# Load idg-util library
lib = ctypes.cdll.LoadLibrary('libidg-util.so')

class Data():
    def __init__(
        self,
        grid_size,
        nr_stations_limit,
        baseline_length_limit,
        layout_file,
        start_frequency):
        lib.DATA_init.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_char_p,
            ctypes.c_float]
        self.obj = lib.DATA_init(
            ctypes.c_uint(grid_size),
            ctypes.c_uint(nr_stations_limit),
            ctypes.c_uint(baseline_length_limit),
            ctypes.c_char_p(layout_file),
            ctypes.c_float(start_frequency))

    def get_image_size(
        self):
        lib.DATA_get_image_size.restype = ctypes.c_float
        return lib.DATA_get_image_size(self.obj)

    def get_nr_stations(
        self):
        lib.DATA_get_nr_stations.restype = ctypes.c_uint
        return lib.DATA_get_nr_stations(self.obj)

    def get_nr_baselines(
        self):
        lib.DATA_get_nr_baselines.restype = ctypes.c_uint
        return lib.DATA_get_nr_baselines(self.obj)

    def get_frequencies(
        self,
        frequencies,
        nr_channels,
        channel_offset):
        lib.DATA_get_frequencies.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint]
        lib.DATA_get_frequencies(
            self.obj,
            frequencies.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint(nr_channels),
            ctypes.c_uint(channel_offset))


    def get_uvw(
        self,
        uvw,
        nr_baselines,
        nr_timesteps,
        baseline_offset,
        time_offset,
        integration_time):
        lib.DATA_get_uvw.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_float]
        lib.DATA_get_uvw(
            self.obj,
            uvw.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint(nr_baselines),
            ctypes.c_uint(nr_timesteps),
            ctypes.c_uint(baseline_offset),
            ctypes.c_uint(time_offset),
            ctypes.c_float(integration_time))
