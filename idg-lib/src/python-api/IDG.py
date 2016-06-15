import os
import ctypes
import numpy
# from enum import Enum  # not supprted on all systems
from ctypes.util import find_library

# A bit ugly, but ctypes.util's find_library does not look in
# the LD_LIBRARY_PATH, but only PATH. Howver, we can also provide
# the full path of the shared object file
path = os.path.dirname(os.path.realpath(__file__))
path, junk = os.path.split(path)
path, junk = os.path.split(path)
libpath = os.path.join(path, 'libidg.so')
lib = ctypes.cdll.LoadLibrary(libpath)


# class Direction(Enum):
class Direction():
    FourierToImage = 0
    ImageToFourier = 1

class Type():
    CPU_REFERENCE = 0
    CPU_OPTIMIZED = 1
    CUDA_KEPLER   = 2
    CUDA_MAXWELL  = 3


class Scheme():

    def get_stations(self):
        """Get the number of stations"""
        return lib.Scheme_get_stations(self.obj)

    def set_stations(self, n):
        """Set the number of stations"""
        lib.Scheme_set_stations(self.obj, ctypes.c_int(n))

    def set_frequencies(self, frequencies):
        """
        Set the frequencies
        :param frequencies: numpy.ndarray(nr_channels, dtype=numpy.float64)
        :type frequencies: numpy.ndarray
        """
        lib.Scheme_set_frequencies(
            self.obj,
            ctypes.c_int(frequencies.shape[0]),
            frequencies.ctypes.data_as(ctypes.c_void_p))

    def get_frequency(self, channel):
        """
        Get the frequency of channel 'channel'
        :param channel: channel 0 to NR_CHANNELS-1
        :type channel: int
        """
        lib.Scheme_get_frequency.restype = ctypes.c_double
        return lib.Scheme_get_frequency(self.obj, ctypes.c_int(channel))

    def get_frequencies_size(self):
        """Get the number of channels"""
        return lib.Scheme_get_frequencies_size(self.obj)

    def set_grid(self, grid):
        """
        Set the grid to which to grid the visibilities
        :param grid: numpy.ndarray((nr_polarizations, height, width), dtype=numpy.complex128)
        :type grid: numpy.ndarray
        """
        lib.Scheme_set_grid(
            self.obj,
            ctypes.c_int(grid.shape[0]),  # nr polarizations
            ctypes.c_int(grid.shape[1]),  # height
            ctypes.c_int(grid.shape[2]),  # witdh
            grid.ctypes.data_as(ctypes.c_void_p))

    def get_nr_polarizations(self):
        """Get the number of polarization pairs, i.e. 1 or 4"""
        return lib.Scheme_get_nr_polarizations(self.obj)

    def get_grid_height(self):
        """Get the grid height in pixel"""
        return lib.Scheme_get_grid_height(self.obj)

    def get_grid_width(self):
        """Get the grid width in pixel"""
        return lib.Scheme_get_grid_width(self.obj)

    def internal_get_subgrid_size(self):
        """Get the subgrid size in pixel"""
        return lib.Scheme_internal_get_subgrid_size(self.obj)

    def internal_set_subgrid_size(self, size):
        """Set the subgrid size in pixel"""
        lib.Scheme_internal_set_subgrid_size(self.obj, ctypes.c_int(size))

    def get_w_kernel_size(self):
        """Get the w-kernel size in pixel"""
        return lib.Scheme_get_w_kernel_size(self.obj)

    def set_w_kernel_size(self, size):
        """Set the w-kernel size in pixel"""
        lib.Scheme_set_w_kernel_size(self.obj, ctypes.c_int(size))

    def set_spheroidal(self, spheroidal):
        """
        Set the grid to which to grid the visibilities
        :param spheroidal: numpy.ndarray((height, width), dtype=numpy.float64)
        :type spheroidal: numpy.ndarray
        """
        lib.Scheme_set_spheroidal(
            self.obj,
            ctypes.c_int(spheroidal.shape[0]),  # height
            ctypes.c_int(spheroidal.shape[1]),  # witdh
            spheroidal.ctypes.data_as(ctypes.c_void_p))

    def set_cell_size(self, height, width):
        """Set the cell size"""
        lib.Scheme_set_cell_size(self.obj,
                                 ctypes.c_double(height),
                                 ctypes.c_double(width))

    def get_cell_height(self):
        """Get the cell height in [unit]"""
        lib.Scheme_get_cell_height.restype = ctypes.c_double
        lib.Scheme_get_cell_height(self.obj)

    def get_cell_width(self):
        """Set the cell width in [unit]"""
        lib.Scheme_get_cell_width.restype = ctypes.c_double
        lib.Scheme_get_cell_width(self.obj)

    # deprecated: use cell size!
    def get_image_size(self):
        """Get the image size"""
        lib.Scheme_get_image_size.restype = ctypes.c_double
        return lib.Scheme_get_image_size(self.obj)

    # deprecated: use cell size!
    def set_image_size(self, size):
        """Get the image size in [unit]"""
        lib.Scheme_set_image_size(self.obj, ctypes.c_double(size))

    def bake(self):
        """Bake the plan after all parameters are set"""
        lib.Scheme_bake(self.obj)

    def flush(self):
        """Flush buffer explicitly"""
        lib.Scheme_flush(self.obj)

    def start_aterm(self, aterms):
        """Start a new A-term
        :param aterms: numpy.ndarray(nr_stations, subgrid_size,
                                     subgrid_size, nr_polarizations),
                                     dtype=numpy.complex128)
        """
        nr_stations      = aterms.shape[0]
        height           = aterms.shape[1]
        width            = aterms.shape[2]
        nr_polarizations = aterms.shape[3]
        lib.Scheme_start_aterm(self.obj,
                               ctypes.c_int(nr_stations),
                               ctypes.c_int(height),
                               ctypes.c_int(width),
                               ctypes.c_int(nr_polarizations),
                               aterms.ctypes.data_as(ctypes.c_void_p))


    def finish_aterm(self):
        lib.Scheme_finish_aterm(self.obj)


    def ifft_grid(self):
        """Inverse FFT on the grid"""
        lib.Scheme_ifft_grid(self.obj)


    def fft_grid(self):
        """FFT on the grid"""
        lib.Scheme_fft_grid(self.obj)


    def get_copy_grid(self):
        """Get a copy of the grid"""
        nr_polarizations = self.get_nr_polarizations()
        height           = self.get_grid_height()
        width            = self.get_grid_width()

        grid = numpy.zeros(shape=(nr_polarizations, height, width),
                           dtype=numpy.complex128)
        lib.Scheme_copy_grid(
            self.obj,
            ctypes.c_int(nr_polarizations),
            ctypes.c_int(height),
            ctypes.c_int(width),
            grid.ctypes.data_as(ctypes.c_void_p))

        return grid



class GridderPlan(Scheme):

    def __init__(self, proxytype, bufferTimesteps):
        """Create a Gridder plan"""
        lib.GridderPlan_init.argtypes = [ctypes.c_uint, ctypes.c_uint]
        self.obj = lib.GridderPlan_init(
            ctypes.c_uint(proxytype),
            ctypes.c_uint(bufferTimesteps)
        )


    def __del__(self):
        """Destroy a Gridder plan"""
        lib.GridderPlan_destroy(self.obj)


    def grid_visibilities(
            self,
            timeIndex,
            antenna1,
            antenna2,
            uvw_coordinates,
            visibilities):
        """
        Place visibilities into the buffer
        """
        lib.GridderPlan_grid_visibilities(
            self.obj,
            ctypes.c_int(timeIndex),
            ctypes.c_int(antenna1),
            ctypes.c_int(antenna2),
            uvw_coordinates.ctypes.data_as(ctypes.c_void_p),
            visibilities.ctypes.data_as(ctypes.c_void_p)
        )


    def transform_grid(self, grid=None, crop_tolarance=0.005):
        """
        Inverse FFT on the grid + apply the spheroidal + scaling,
        including setting the imaginary part to zero
        """
        if (grid == None):
            null_ptr = ctypes.POINTER(ctypes.c_int)()
            lib.DegridderPlan_transform_grid(
                self.obj,
                ctypes.c_double(crop_tolarance),
                ctypes.c_int(0),
                ctypes.c_int(0),
                ctypes.c_int(0),
                null_ptr)
        else:
            lib.GridderPlan_transform_grid(
                self.obj,
                ctypes.c_double(crop_tolarance),
                ctypes.c_int(grid.shape[0]),
                ctypes.c_int(grid.shape[1]),
                ctypes.c_int(grid.shape[2]),
                grid.ctypes.data_as(ctypes.c_void_p))



class DegridderPlan(Scheme):

    def __init__(self, proxytype, bufferTimesteps):
        """Create a Gridder plan"""
        lib.DegridderPlan_init.argtypes = [ctypes.c_uint, ctypes.c_uint]
        self.obj = lib.DegridderPlan_init(
            ctypes.c_uint(proxytype),
            ctypes.c_uint(bufferTimesteps)
        )

    def __del__(self):
        """Destroy a Gridder plan"""
        lib.DegridderPlan_destroy(self.obj)

    def request_visibilities(self,
                             timeIndex,
                             antenna1,
                             antenna2,
                             uvw):
        """Request visibilities to be put into the buffer"""
        lib.DegridderPlan_request_visibilities(
            self.obj,
            ctypes.c_int(timeIndex),
            ctypes.c_int(antenna1),
            ctypes.c_int(antenna2),
            uvw.ctypes.data_as(ctypes.c_void_p))


    def read_visibilities(self, timeIndex, antenna1, antenna2):
        """Read visibilities from the buffer"""
        nr_channels = self.get_frequencies_size()
        nr_polarizations = self.get_nr_polarizations()
        visibilities =  numpy.zeros((nr_channels, nr_polarizations),
                                    dtype=numpy.complex64)
        lib.DegridderPlan_read_visibilities(self.obj,
                                            ctypes.c_int(timeIndex),
                                            ctypes.c_int(antenna1),
                                            ctypes.c_int(antenna2),
                                            visibilities.ctypes.data_as(ctypes.c_void_p))
        return visibilities


    def transform_grid(self, grid=None, crop_tolarance=0.005):
        """Do an FFT on the grid and apply the spheroidal"""
        if (grid == None):
            null_ptr = ctypes.POINTER(ctypes.c_int)()
            lib.DegridderPlan_transform_grid(
                self.obj,
                ctypes.c_double(crop_tolarance),
                ctypes.c_int(0),
                ctypes.c_int(0),
                ctypes.c_int(0),
                null_ptr)
        else:
            lib.DegridderPlan_transform_grid(
                self.obj,
                ctypes.c_double(crop_tolarance),
                ctypes.c_int(grid.shape[0]),
                ctypes.c_int(grid.shape[1]),
                ctypes.c_int(grid.shape[2]),
                grid.ctypes.data_as(ctypes.c_void_p))
