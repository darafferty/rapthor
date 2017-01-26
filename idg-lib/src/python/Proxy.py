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

    def gridding(
        self,
        w_offset,
        cell_size,
        kernel_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        grid,
        aterms,
        aterms_offset,
        spheroidal):
        """
        Grid visibilities onto grid.

        :param frequencies: numpy.ndarray(
                shapenr_channels,
                dtype = idg.frequenciestype)
        :param visibilities: numpy.ndarray(
                shape=(nr_baselines, nr_timesteps, nr_channels, nr_correlations),
                dtype=idg.visibilitiestype)
        :param uvw: numpy.ndarray(
                shape=(nr_baselines, nr_timesteps),
                dtype = idg.uvwtype)
        :param baselines: numpy.ndarray(
                shape=(nr_baselines),
                dtype=idg.baselinetype)
        :param grid: numpy.ndarray(
                shape=(nr_correlations, height, width),
                dtype = idg.gridtype)
        :param aterms: numpy.ndarray(
                shape=(nr_timeslots, nr_stations, height, width, nr_correlations),
                dtype = idg.atermtype)
        :param aterms_offset: numpy.ndarray(
                shape=(nr_timeslots+1),
                dtype = idg.atermoffsettype)
        :param spheroidal: numpy.ndarray(
                shape=(height, width),
                dtype = idg.spheroidaltype)
        """
        # extract dimensions
        nr_channels = frequencies.shape[0]
        visibilities_nr_baselines    = visibilities.shape[0]
        visibilities_nr_timesteps    = visibilities.shape[1]
        visibilities_nr_channels     = visibilities.shape[2]
        visibilities_nr_correlations = visibilities.shape[3]
        uvw_nr_baselines             = uvw.shape[0]
        uvw_nr_timesteps             = uvw.shape[1]
        uvw_nr_coordinates           = uvw.shape[2]
        baselines_nr_baselines       = baselines.shape[0]
        baselines_two                = baselines.shape[1]
        grid_nr_correlations         = grid.shape[0]
        grid_height                  = grid.shape[0]
        grid_width                   = grid.shape[0]
        aterms_nr_timeslots          = aterms.shape[0]
        aterms_nr_stations           = aterms.shape[1]
        aterms_aterm_height          = aterms.shape[2]
        aterms_aterm_width           = aterms.shape[3]
        aterms_nr_correlations       = aterms.shape[4]
        aterms_offsets_nr_timeslots  = aterms_offset.shape[0]
        spheroidal_height            = spheroidal.shape[0]
        spheroidal_width             = spheroidal.shape[1]

        # call C function to do the work
        self._cwrap_griddding(
            w_offset,
            cell_size,
            kernel_size,
            frequencies,
            nr_channels,
            visibilities,
            visibilities_nr_baselines,
            visibilities_nr_timesteps,
            visibilities_nr_channels,
            visibilities_nr_correlations,
            uvw,
            uvw_nr_baselines,
            uvw_nr_timesteps,
            uvw_nr_coordinates,
            baselines,
            baselines_nr_baselines,
            baselines_two,
            grid,
            grid_nr_correlations,
            grid_height,
            grid_width,
            aterms,
            aterms_nr_timeslots,
            aterms_nr_stations,
            aterms_aterm_height,
            aterms_aterm_width,
            aterms_nr_correlations,
            aterms_offsets,
            aterms_offsets_nr_timeslots, # plus one
            spheroidal,
            spheroidal_height,
            spheroidal_width)


    def degridding(
        self,
        w_offset,
        cell_size,
        kernel_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        grid,
        aterms,
        aterms_offset,
        spheroidal):
        """
        Degrid visibilities from grid.

        :param frequencies: numpy.ndarray(
                shapenr_channels,
                dtype = idg.frequenciestype)
        :param visibilities: numpy.ndarray(
                shape=(nr_baselines, nr_timesteps, nr_channels, nr_correlations),
                dtype=idg.visibilitiestype)
        :param uvw: numpy.ndarray(
                shape=(nr_baselines, nr_timesteps),
                dtype = idg.uvwtype)
        :param baselines: numpy.ndarray(
                shape=(nr_baselines),
                dtype=idg.baselinetype)
        :param grid: numpy.ndarray(
                shape=(nr_correlations, height, width),
                dtype = idg.gridtype)
        :param aterms: numpy.ndarray(
                shape=(nr_timeslots, nr_stations, height, width, nr_correlations),
                dtype = idg.atermtype)
        :param aterms_offset: numpy.ndarray(
                shape=(nr_timeslots+1),
                dtype = idg.atermoffsettype)
        :param spheroidal: numpy.ndarray(
                shape=(height, width),
                dtype = idg.spheroidaltype)
        """
        # extract dimensions
        nr_channels = frequencies.shape[0]
        visibilities_nr_baselines    = visibilities.shape[0]
        visibilities_nr_timesteps    = visibilities.shape[1]
        visibilities_nr_channels     = visibilities.shape[2]
        visibilities_nr_correlations = visibilities.shape[3]
        uvw_nr_baselines             = uvw.shape[0]
        uvw_nr_timesteps             = uvw.shape[1]
        uvw_nr_coordinates           = uvw.shape[2]
        baselines_nr_baselines       = baselines.shape[0]
        baselines_two                = baselines.shape[1]
        grid_nr_correlations         = grid.shape[0]
        grid_height                  = grid.shape[0]
        grid_width                   = grid.shape[0]
        aterms_nr_timeslots          = aterms.shape[0]
        aterms_nr_stations           = aterms.shape[1]
        aterms_aterm_height          = aterms.shape[2]
        aterms_aterm_width           = aterms.shape[3]
        aterms_nr_correlations       = aterms.shape[4]
        aterms_offsets_nr_timeslots  = aterms_offset.shape[0]
        spheroidal_height            = spheroidal.shape[0]
        spheroidal_width             = spheroidal.shape[1]

        # call C function to do the work
        self._cwrap_degriddding(
            w_offset,
            cell_size,
            kernel_size,
            frequencies,
            nr_channels,
            visibilities,
            visibilities_nr_baselines,
            visibilities_nr_timesteps,
            visibilities_nr_channels,
            visibilities_nr_correlations,
            uvw,
            uvw_nr_baselines,
            uvw_nr_timesteps,
            uvw_nr_coordinates,
            baselines,
            baselines_nr_baselines,
            baselines_two,
            grid,
            grid_nr_correlations,
            grid_height,
            grid_width,
            aterms,
            aterms_nr_timeslots,
            aterms_nr_stations,
            aterms_aterm_height,
            aterms_aterm_width,
            aterms_nr_correlations,
            aterms_offsets,
            aterms_offsets_nr_timeslots, # plus one
            spheroidal,
            spheroidal_height,
            spheroidal_width)


    def transform(
        self,
        direction,
        grid):
        """
        Transform Fourier Domain<->Image Domain.

        :param direction: idg.FourierDomainToImageDomain or idg.ImageDomainToFourierDomain
        :param grid: numpy.ndarray(
                shape=(nr_correlations, height, width),
                dtype = idg.gridtype)
        """
        # extract dimesions
        nr_correlations = grid.shape[0]
        height          = grid.shape[0]
        width           = grid.shape[0]

        # call C function to do the work
        self._cwrap_transform(
            direction,
            grid,
            nr_correlations,
            grid_height,
            grid_width)

 
    # TODO: create plan

    # Wrapper to C function (override for each class inheriting from this)
    def _cwrap_gridding(
        self,
        w_offset,
        cell_size,
        kernel_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        grid,
        aterms,
        aterms_offset,
        spheroidal):
        pass

    # Wrapper to C function (override for each class inheriting from this)
    def _cwrap_degridding(
        self,
        w_offset,
        cell_size,
        kernel_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        grid,
        aterms,
        aterms_offset,
        spheroidal):
        pass

    # Wrapper to C function (override for each class inheriting from this)
    def _cwrap_transform(
        self,
        direction,
        grid):
        pass
