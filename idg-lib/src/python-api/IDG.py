import os
import ctypes
import numpy
from ctypes.util import find_library

# A bit ugly, but ctypes.util's find_library does not look in
# the LD_LIBRARY_PATH, but only PATH. Howver, we can also provide
# the full path of the shared object file
path = os.path.dirname(os.path.realpath(__file__))
path, junk = os.path.split(path)
path, junk = os.path.split(path)
libpath = os.path.join(path, 'libidg.so')
lib = ctypes.cdll.LoadLibrary(libpath)


class GridderPlan():

    def __init__(self, bufferTimesteps):
        """Create a Gridder plan"""
        lib.GridderPlan_init.argtypes = [ctypes.c_uint]
        self.obj = lib.GridderPlan_init(
            ctypes.c_uint(bufferTimesteps)
        )

    def __del__(self):
        """Destroy a Gridder plan"""
        lib.GridderPlan_destroy(self.obj)

    def get_stations(self):
        """Get the number of stations"""
        return lib.GridderPlan_get_stations(self.obj)

    def set_stations(self, n):
        """Set the number of stations"""
        lib.GridderPlan_set_stations(self.obj, ctypes.c_int(n))

    def set_frequencies(self, frequencies):
        """
        Set the frequencies
        :param frequencies: numpy.ndarray(nr_channels, dtype=numpy.float64)
        :type frequencies: numpy.ndarray
        """
        lib.GridderPlan_set_frequencies(
            self.obj,
            frequencies.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(frequencies.shape[0])
        )

    def get_frequency(self, channel):
        """
        Get the frequency of channel 'channel'
        :param channel: channel 0 to NR_CHANNELS-1
        :type channel: int
        """
        lib.GridderPlan_get_frequency.restype = ctypes.c_double
        return lib.GridderPlan_get_frequency(self.obj, ctypes.c_int(channel))

    def get_frequencies_size(self):
        """Get the number of channels"""
        return lib.GridderPlan_get_frequencies_size(self.obj)

    def set_grid(self, grid):
        """
        Set the grid to which to grid the visibilities
        :param grid: numpy.ndarray((nr_polarizations, height, width), dtype=numpy.complex128)
        :type grid: numpy.ndarray
        """
        lib.GridderPlan_set_grid(
            self.obj,
            grid.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(grid.shape[0]),  # nr polarizations
            ctypes.c_int(grid.shape[1]),  # height
            ctypes.c_int(grid.shape[2])   # witdh
        )

    def get_subgrid_size(self):
        """Get the subgrid size"""
        return lib.GridderPlan_get_subgrid_size(self.obj)

    def set_subgrid_size(self, size):
        """Set the subgrid size"""
        lib.GridderPlan_set_subgrid_size(self.obj, ctypes.c_int(size))

    def get_w_kernel_size(self):
        """Get the w-kernel size"""
        return lib.GridderPlan_get_w_kernel_size(self.obj)

    def set_w_kernel_size(self, size):
        """Set the w-kernel size"""
        lib.GridderPlan_set_w_kernel_size(self.obj, ctypes.c_int(size))

    def set_spheroidal(self, spheroidal):
        """
        Set the grid to which to grid the visibilities
        :param spheroidal: numpy.ndarray((height, width), dtype=numpy.float64)
        :type spheroidal: numpy.ndarray
        """
        lib.GridderPlan_set_spheroidal(
            self.obj,
            spheroidal.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(spheroidal.shape[0]),  # height
            ctypes.c_int(spheroidal.shape[1])   # witdh
        )

    # deprecated: use cell size!
    def get_image_size(self):
        """Get the w-kernel size"""
        lib.GridderPlan_get_image_size.restype = ctypes.c_double
        return lib.GridderPlan_get_image_size(self.obj)

    # deprecated: use cell size!
    def set_image_size(self, size):
        """Set the w-kernel size"""
        lib.GridderPlan_set_image_size(self.obj, ctypes.c_double(size))

    def bake(self):
        """Bake the plan after all parameters are set"""
        lib.GridderPlan_bake(self.obj)

    def start_aterm(self, aterms):
        """Start a new A-term
        :param aterms: numpy.ndarray(nr_stations, subgrid_size,
                                     subgrid_size, nr_polarizations),
                                     dtype=numpy.complex128)
        """
        nr_stations = aterms.shape[0]
        height = aterms.shape[1]
        width = aterms.shape[2]
        nr_polarizations = aterms.shape[3]
        lib.GridderPlan_start_aterm(self.obj,
                                    aterms.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(nr_stations),
                                    ctypes.c_int(height),
                                    ctypes.c_int(width),
                                    ctypes.c_int(nr_polarizations))


    def finish_aterm(self):
        lib.GridderPlan_finish_aterm(self.obj)


    def grid_visibilities(
            self,
            visibilities,
            uvw_coordinates,
            antenna1,
            antenna2,
            timeIndex):
        """
        Place visibilities into the buffer
        :param spheroidal: numpy.ndarray(nr_channels, nr_polarizations),
                           dtype=numpy.float32)
        :type spheroidal: numpy.ndarray
        """
        lib.GridderPlan_grid_visibilities(
            self.obj,
            visibilities.ctypes.data_as(ctypes.c_void_p),
            uvw_coordinates.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(antenna1),
            ctypes.c_int(antenna2),
            ctypes.c_int(timeIndex)
        )

    def flush(self):
        """Flush buffer"""
        lib.GridderPlan_flush(self.obj)
