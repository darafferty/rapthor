import os
import ctypes

# A bit ugly, but ctypes.util's find_library does not look in
# the LD_LIBRARY_PATH, but only PATH. Howver, we can also provide
# the full path of the shared object file
path = os.path.dirname(os.path.realpath(__file__))
path, junk = os.path.split(path)
path, junk = os.path.split(path)
libpath = os.path.join(path, 'libidg-common.so')
lib = ctypes.cdll.LoadLibrary(libpath)


class Proxy(object):

    def grid_visibilities(self,
                          visibilities,
                          uvw,
                          wavenumbers,
                          metadata,
                          grid,
                          w_offset,
                          aterms,
                          spheroidal):
        """Grid visibilities onto grid.

        Arguments:
        visibilities - numpy.ndarray(shape=(nr_baselines, nr_time,
                       nr_channels, nr_polarizations),
                       dtype=idg.visibilitiestype)
        uvw - numpy.ndarray(shape=(nr_baselines, nr_time),
                            dtype = idg.uvwtype)
        wavenumbers - numpy.ndarray(nr_channels, dtype = idg.wavenumberstype)
        metadata - numpy.ndarray(nr_baselines, nr_timeslots, dtype=idg.metadatatype)
        grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
                             dtype = idg.gridtype)
        aterms - numpy.ndarray(shape=(nr_stations, nr_timeslots,
                               nr_polarizations, subgrid_size, subgrid_size),
                               dtype = idg.atermtype)
        spheroidal - numpy.ndarray(shape=(subgrid_size, subgrid_size),
                                   dtype = idg.spheroidaltype)
        """
        # check dimensions
        if visibilities.shape != (self.get_nr_baselines(),
                                  self.get_nr_time(),
                                  self.get_nr_channels(),
                                  self.get_nr_polarizations()):
            raise ValueError('Visibilities dimension missmatch.')
        if uvw.shape != (self.get_nr_baselines(),
                         self.get_nr_time()):
            raise ValueError('UVW dimension missmatch.')
        if wavenumbers.shape != (self.get_nr_channels(), ):
            raise ValueError('Wavenumbers dimension missmatch.')
        if metadata.shape != (self.get_nr_baselines(), self.get_nr_timeslots()):
            raise ValueError('Metadata dimension missmatch.')
        if grid.shape != (self.get_nr_polarizations(),
                          self.get_grid_size(),
                          self.get_grid_size()):
            raise ValueError('Grid dimension missmatch.')
        if aterms.shape != (self.get_nr_stations(),
                            self.get_nr_timeslots(),
                            self.get_nr_polarizations(),
                            self.get_subgrid_size(),
                            self.get_subgrid_size()):
            raise ValueError('Aterms dimension missmatch.')
        if spheroidal.shape != (self.get_subgrid_size(),
                                self.get_subgrid_size()):
            raise ValueError('Spheroidal dimension missmatch.')

        # call C function to do the work
        self._cwrap_grid_visibilities(visibilities, uvw, wavenumbers,
                                      metadata, grid, w_offset, aterms,
                                      spheroidal)


    def degrid_visibilities(self,
                            visibilities,
                            uvw,
                            wavenumbers,
                            metadata,
                            grid,
                            w_offset,
                            aterms,
                            spheroidal):
        """Degrid visibilities onto grid.

        Arguments:
        visibilities - numpy.ndarray(shape=(nr_baselines, nr_time,
                                     nr_channels, nr_polarizations),
        dtype=idg.visibilitiestype)
        uvw - numpy.ndarray(shape=(nr_baselines,nr_time),
                            dtype = idg.uvwtype)
        wavenumbers - numpy.ndarray(nr_channels, dtype = idg.wavenumberstype)
        metadata - numpy.ndarray(nr_baselines, nr_timeslots, dtype=idg.metadatatype)
        grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
        dtype = idg.gridtype)
        aterms - numpy.ndarray(shape=(nr_stations, nr_timeslots,
                 nr_polarizations, subgrid_size, subgrid_size),
                 dtype = idg.atermtype)
        spheroidal - numpy.ndarray(shape=(subgrid_size, subgrid_size),
        dtype = idg.spheroidaltype)
        """
        # check dimensions
        if visibilities.shape != (self.get_nr_baselines(),
                                  self.get_nr_time(),
                                  self.get_nr_channels(),
                                  self.get_nr_polarizations()):
            raise ValueError('Visibilities dimension missmatch.')
        if uvw.shape != (self.get_nr_baselines(),
                         self.get_nr_time()):
            raise ValueError('UVW dimension missmatch.')
        if wavenumbers.shape != (self.get_nr_channels(), ):
            raise ValueError('Wavenumbers dimension missmatch.')
        if metadata.shape != (self.get_nr_baselines(), self.get_nr_timeslots()):
            raise ValueError('Metadata dimension missmatch.')
        if grid.shape != (self.get_nr_polarizations(),
                          self.get_grid_size(),
                          self.get_grid_size()):
            raise ValueError('Grid dimension missmatch.')
        if aterms.shape != (self.get_nr_stations(),
                            self.get_nr_timeslots(),
                            self.get_nr_polarizations(),
                            self.get_subgrid_size(),
                            self.get_subgrid_size()):
            raise ValueError('Aterms dimension missmatch.')
        if spheroidal.shape != (self.get_subgrid_size(),
                                self.get_subgrid_size()):
            raise ValueError('Spheroidal dimension missmatch.')

        # call C function to do the work
        self._cwrap_degrid_visibilities(visibilities, uvw, wavenumbers,
                                        metadata, grid, w_offset, aterms,
                                        spheroidal)



    def transform(self,
                  direction,
                  grid):
        """Transform Fourier Domain<->Image Domain.

        Arguments:
        direction - idg.FourierDomainToImageDomain or idg.ImageDomainToFourierDomain
        grid - numpy.ndarray(shape=(nr_polarizations, grid_size, grid_size),
        dtype = idg.gridtype)
        """
        # check argument dimesions
        if grid.shape != (self.get_nr_polarizations(),
                          self.get_grid_size(),
                          self.get_grid_size()):
            raise ValueError('Grid dimension missmatch.')

        # call C function to do the work
        self._cwrap_transform(direction, grid)



    def get_nr_stations(self):
        return lib.Proxy_get_nr_stations(self.obj)

    def get_nr_baselines(self):
        return lib.Proxy_get_nr_baselines(self.obj)

    def get_nr_channels(self):
        return lib.Proxy_get_nr_channels(self.obj)

    def get_nr_time(self):
        return lib.Proxy_get_nr_time(self.obj)

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


    # Wrapper to C function (override for each class inheriting from this)
    def _cwrap_grid_visibilities(self, visibilities, uvw, wavenumbers,
                                metadata, grid, w_offset, aterms,
                                spheroidal):
        pass

    # Wrapper to C function (override for each class inheriting from this)
    def _cwrap_degrid_visibilities(self, visibilities, uvw,
                                   wavenumbers, metadata,
                                   grid, w_offset, aterms,
                                   spheroidal):
        pass

    # Wrapper to C function (override for each class inheriting from this)
    def _cwrap_transform(self, direction, grid):
        pass
