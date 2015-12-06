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
libpath = os.path.join(path, 'libidg.so')
lib = ctypes.cdll.LoadLibrary(libpath)



class Reference(Proxy):
    """Reference CPU implementation"""
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
        """Grid visibilities onto grid.

        Arguments:
        visibilities - numpy.ndarray(shape=(nr_subgrids, nr_timesteps,
                                            nr_channels, nr_polarizations),
                                            dtype=idg.visibilitiestype)
        uvw - numpy.ndarray(shape=(nr_timesteps, 3), dtype = idg.uvwtype)
        wavenumbers - numpy.ndarray(nr_channels, dtype = idg.wavenumberstype)
        metadata - numpy.ndarray(nr_subgrids, dtype=idg.metadatatype)
        grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
                             dtype = idg.gridtype)
        aterm - numpy.ndarray(shape=(nr_stations, nr_timeslots, nr_polarizations,
                             subgrid_size, subgrid_size), dtype = idg.atermtype)
        spheroidal - numpy.ndarray(shape=(subgrid_size, subgrid_size),
                     dtype = idg.spheroidaltype)
        """
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
        """Degrid visibilities onto grid.

        Arguments:
        visibilities - numpy.ndarray(shape=(nr_subgrids, nr_timesteps,
                                            nr_channels, nr_polarizations),
                                            dtype=idg.visibilitiestype)
        uvw - numpy.ndarray(shape=(nr_timesteps, 3), dtype = idg.uvwtype)
        wavenumbers - numpy.ndarray(nr_channels, dtype = idg.wavenumberstype)
        metadata - numpy.ndarray(nr_subgrids, dtype=idg.metadatatype)
        grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
                             dtype = idg.gridtype)
        aterm - numpy.ndarray(shape=(nr_stations, nr_timeslots, nr_polarizations,
                             subgrid_size, subgrid_size), dtype = idg.atermtype)
        spheroidal - numpy.ndarray(shape=(subgrid_size, subgrid_size),
                     dtype = idg.spheroidaltype)
        """
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
        """Transform Fourier Domain<->Image Domain.

        Arguments:
        direction - idg.FourierDomainToImageDomain or idg.ImageDomainToFourierDomain
        grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
                             dtype = idg.gridtype)
        """
        lib.CPU_Reference_transform(self.obj,
                                    ctypes.c_int(direction),
                                    grid.ctypes.data_as(ctypes.c_void_p))

    def get_job_size_gridder(self):
        return lib.CPU_get_job_size_gridder(self.obj)

    def get_job_size_adder(self):
        return lib.CPU_get_job_size_adder(self.obj)

    def get_job_size_splitter(self):
        return lib.CPU_get_job_size_splitter(self.obj)

    def get_job_size_degridder(self):
        return lib.CPU_get_job_size_degridder(self.obj)

    def set_job_size_gridder(self, n = 8192):
        lib.CPU_set_job_size_gridder(self.obj, ctypes.c_int(n))

    def set_job_size_adder(self, n = 8192):
        lib.CPU_set_job_size_adder(self.obj, ctypes.c_int(n))

    def set_job_size_splitter(self, n = 8192):
        lib.CPU_set_job_size_splitter(self.obj, ctypes.c_int(n))

    def set_job_size_degridder(self, n = 8192):
        lib.CPU_set_job_size_degridder(self.obj, ctypes.c_int(n))



class SandyBridgeEP(Reference):
    """CPU implementation optimized for Intel SandyBridgeEP"""
    def __init__(self, nr_stations,
                       nr_channels,
                       nr_timesteps,
                       nr_timeslots,
                       imagesize,
                       grid_size,
                       subgrid_size = 32):
        try:
            lib.CPU_SandyBridgeEP_init.argtypes = [ctypes.c_uint, \
                                                   ctypes.c_uint, \
                                                   ctypes.c_uint, \
                                                   ctypes.c_uint, \
                                                   ctypes.c_float, \
                                                   ctypes.c_uint, \
                                                   ctypes.c_uint]
            self.obj = lib.CPU_SandyBridgeEP_init(
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
        """Grid visibilities onto grid.

        Arguments:
        visibilities - numpy.ndarray(shape=(nr_subgrids, nr_timesteps,
                                            nr_channels, nr_polarizations),
                                            dtype=idg.visibilitiestype)
        uvw - numpy.ndarray(shape=(nr_timesteps, 3), dtype = idg.uvwtype)
        wavenumbers - numpy.ndarray(nr_channels, dtype = idg.wavenumberstype)
        metadata - numpy.ndarray(nr_subgrids, dtype=idg.metadatatype)
        grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
                             dtype = idg.gridtype)
        aterm - numpy.ndarray(shape=(nr_stations, nr_timeslots, nr_polarizations,
                             subgrid_size, subgrid_size), dtype = idg.atermtype)
        spheroidal - numpy.ndarray(shape=(subgrid_size, subgrid_size),
                     dtype = idg.spheroidaltype)
        """
        lib.CPU_SandyBridgeEP_grid(self.obj,
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
        """Degrid visibilities onto grid.

        Arguments:
        visibilities - numpy.ndarray(shape=(nr_subgrids, nr_timesteps,
                                            nr_channels, nr_polarizations),
                                            dtype=idg.visibilitiestype)
        uvw - numpy.ndarray(shape=(nr_timesteps, 3), dtype = idg.uvwtype)
        wavenumbers - numpy.ndarray(nr_channels, dtype = idg.wavenumberstype)
        metadata - numpy.ndarray(nr_subgrids, dtype=idg.metadatatype)
        grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
                             dtype = idg.gridtype)
        aterm - numpy.ndarray(shape=(nr_stations, nr_timeslots, nr_polarizations,
                             subgrid_size, subgrid_size), dtype = idg.atermtype)
        spheroidal - numpy.ndarray(shape=(subgrid_size, subgrid_size),
                     dtype = idg.spheroidaltype)
        """
        lib.CPU_SandyBridgeEP_degrid(self.obj,
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
        """Transform Fourier Domain<->Image Domain.

        Arguments:
        direction - idg.FourierDomainToImageDomain or idg.ImageDomainToFourierDomain
        grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
                             dtype = idg.gridtype)
        """
        lib.CPU_SandyBridgeEP_transform(self.obj,
                                        ctypes.c_int(direction),
                                        grid.ctypes.data_as(ctypes.c_void_p))



class HaswellEP(Reference):
    """CPU implementation optimized for Intel HaswellEP"""
    def __init__(self, nr_stations,
                       nr_channels,
                       nr_timesteps,
                       nr_timeslots,
                       imagesize,
                       grid_size,
                       subgrid_size = 32):
        try:
            lib.CPU_HaswellEP_init.argtypes = [ctypes.c_uint, \
                                               ctypes.c_uint, \
                                               ctypes.c_uint, \
                                               ctypes.c_uint, \
                                               ctypes.c_float, \
                                               ctypes.c_uint, \
                                               ctypes.c_uint]
            self.obj = lib.CPU_HaswellEP_init(
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
        """Grid visibilities onto grid.

        Arguments:
        visibilities - numpy.ndarray(shape=(nr_subgrids, nr_timesteps,
                                            nr_channels, nr_polarizations),
                                            dtype=idg.visibilitiestype)
        uvw - numpy.ndarray(shape=(nr_timesteps, 3), dtype = idg.uvwtype)
        wavenumbers - numpy.ndarray(nr_channels, dtype = idg.wavenumberstype)
        metadata - numpy.ndarray(nr_subgrids, dtype=idg.metadatatype)
        grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
                             dtype = idg.gridtype)
        aterm - numpy.ndarray(shape=(nr_stations, nr_timeslots, nr_polarizations,
                             subgrid_size, subgrid_size), dtype = idg.atermtype)
        spheroidal - numpy.ndarray(shape=(subgrid_size, subgrid_size),
                     dtype = idg.spheroidaltype)
        """
        lib.CPU_HaswellEP_grid(self.obj,
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
        """Degrid visibilities onto grid.

        Arguments:
        visibilities - numpy.ndarray(shape=(nr_subgrids, nr_timesteps,
                                            nr_channels, nr_polarizations),
                                            dtype=idg.visibilitiestype)
        uvw - numpy.ndarray(shape=(nr_timesteps, 3), dtype = idg.uvwtype)
        wavenumbers - numpy.ndarray(nr_channels, dtype = idg.wavenumberstype)
        metadata - numpy.ndarray(nr_subgrids, dtype=idg.metadatatype)
        grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
                             dtype = idg.gridtype)
        aterm - numpy.ndarray(shape=(nr_stations, nr_timeslots, nr_polarizations,
                             subgrid_size, subgrid_size), dtype = idg.atermtype)
        spheroidal - numpy.ndarray(shape=(subgrid_size, subgrid_size),
                     dtype = idg.spheroidaltype)
        """
        lib.CPU_HaswellEP_degrid(self.obj,
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
        """Transform Fourier Domain<->Image Domain.

        Arguments:
        direction - idg.FourierDomainToImageDomain or idg.ImageDomainToFourierDomain
        grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
                             dtype = idg.gridtype)
        """
        lib.CPU_HaswellEP_transform(self.obj,
                                    ctypes.c_int(direction),
                                    grid.ctypes.data_as(ctypes.c_void_p))
