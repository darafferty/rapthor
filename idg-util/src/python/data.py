import numpy
import ctypes
import numpy.ctypeslib

from idg.idgtypes import *

# Load idg-util library
lib = ctypes.cdll.LoadLibrary('libidg-util.so')

class Data():
    def __init__(
        self,
        layout_file):
        lib.DATA_init.restype = ctypes.c_void_p
        lib.DATA_init.argtypes = [
            ctypes.c_char_p]
        self.obj = lib.DATA_init(
            ctypes.c_char_p(layout_file.encode('utf-8')))

    def compute_image_size(
        self,
        grid_size):
        lib.DATA_compute_image_size.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint]
        lib.DATA_compute_image_size.restype = ctypes.c_float
        return lib.DATA_compute_image_size(self.obj, grid_size)

    def compute_max_uv(
        self,
        grid_size):
        lib.DATA_compute_max_uv.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong]
        lib.DATA_compute_max_uv.restype = ctypes.c_float
        return lib.DATA_compute_max_uv(self.obj, grid_size)

    def compute_grid_size(
        self):
        lib.DATA_compute_grid_size.argtypes = [
            ctypes.c_void_p]
        lib.DATA_compute_grid_size.restype = ctypes.c_uint
        return lib.DATA_compute_grid_size(self.obj)

    def limit_max_baseline_length(
        self,
        max_uv):
        lib.DATA_limit_max_baseline_length.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float]
        lib.DATA_limit_max_baseline_length(
            self.obj,
            ctypes.c_float(max_uv))

    def limit_nr_baselines(
        self,
        n):
        lib.DATA_limit_nr_baselines.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint]
        lib.DATA_limit_nr_baselines(
            self.obj,
            ctypes.c_uint(n))

    def limit_nr_stations(
        self,
        n):
        lib.DATA_limit_nr_stations.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint]
        lib.DATA_limit_nr_stations(
            self.obj,
            ctypes.c_uint(n))

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
        image_size,
        channel_offset):
        lib.DATA_get_frequencies.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_float,
            ctypes.c_uint]
        lib.DATA_get_frequencies(
            self.obj,
            frequencies.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint(nr_channels),
            ctypes.c_float(image_size),
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

    def print_info(
        self):
        lib.DATA_print_info.argtypes = [
            ctypes.c_void_p]
        lib.DATA_print_info(self.obj)
