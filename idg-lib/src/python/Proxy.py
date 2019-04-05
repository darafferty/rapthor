import os
import ctypes
import numpy as np
import idg

lib = idg.load_library('libidg-common.so')

class Proxy(object):

    def gridding(
        self,
        w_step,
        shift,
        cell_size,
        kernel_size,
        subgrid_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        grid,
        aterms,
        aterms_offsets,
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
        :param aterms_offsets: numpy.ndarray(
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
        uvw_nr_coordinates           = 3
        baselines_nr_baselines       = baselines.shape[0]
        baselines_two                = 2
        grid_nr_correlations         = grid.shape[0]
        grid_height                  = grid.shape[1]
        grid_width                   = grid.shape[2]
        aterms_nr_timeslots          = aterms.shape[0]
        aterms_nr_stations           = aterms.shape[1]
        aterms_aterm_height          = aterms.shape[2]
        aterms_aterm_width           = aterms.shape[3]
        aterms_nr_correlations       = aterms.shape[4]
        aterms_offsets_nr_timeslots  = aterms_offsets.shape[0]
        spheroidal_height            = spheroidal.shape[0]
        spheroidal_width             = spheroidal.shape[1]

        # call C function to do the work
        self._cwrap_griddding(
            w_step,
            shift,
            cell_size,
            kernel_size,
            subgrid_size,
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
        w_step,
        shift,
        cell_size,
        kernel_size,
        subgrid_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        grid,
        aterms,
        aterms_offsets,
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
        :param aterms_offsets: numpy.ndarray(
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
        uvw_nr_coordinates           = 3
        baselines_nr_baselines       = baselines.shape[0]
        baselines_two                = 2
        grid_nr_correlations         = grid.shape[0]
        grid_height                  = grid.shape[1]
        grid_width                   = grid.shape[2]
        aterms_nr_timeslots          = aterms.shape[0]
        aterms_nr_stations           = aterms.shape[1]
        aterms_aterm_height          = aterms.shape[2]
        aterms_aterm_width           = aterms.shape[3]
        aterms_nr_correlations       = aterms.shape[4]
        aterms_offsets_nr_timeslots  = aterms_offsets.shape[0]
        spheroidal_height            = spheroidal.shape[0]
        spheroidal_width             = spheroidal.shape[1]

        # call C function to do the work
        self._cwrap_degridding(
            w_step,
            shift,
            cell_size,
            kernel_size,
            subgrid_size,
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

    def calibrate(
        self,
        w_step,
        shift,
        cell_size,
        kernel_size,
        subgrid_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        grid,
        aterms,
        aterms_offsets,
        spheroidal):
        """
        Calibrate

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
        :param aterms_offsets: numpy.ndarray(
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
        uvw_nr_coordinates           = 3
        baselines_nr_baselines       = baselines.shape[0]
        baselines_two                = 2
        grid_nr_correlations         = grid.shape[0]
        grid_height                  = grid.shape[1]
        grid_width                   = grid.shape[2]
        aterms_nr_timeslots          = aterms.shape[0]
        aterms_nr_stations           = aterms.shape[1]
        aterms_aterm_height          = aterms.shape[2]
        aterms_aterm_width           = aterms.shape[3]
        aterms_nr_correlations       = aterms.shape[4]
        aterms_offsets_nr_timeslots  = aterms_offsets.shape[0]
        spheroidal_height            = spheroidal.shape[0]
        spheroidal_width             = spheroidal.shape[1]

        # call C function to do the work
        self._cwrap_calibrate(
            w_step,
            shift,
            cell_size,
            kernel_size,
            subgrid_size,
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
        height          = grid.shape[1]
        width           = grid.shape[2]

        # call C function to do the work
        self._cwrap_transform(
            direction,
            grid,
            nr_correlations,
            height,
            width)

    # Wrapper to C function (override for each class inheriting from this)
    def _cwrap_gridding(
        w_step,
        shift,
        cell_size,
        kernel_size,
        subgrid_size,
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
        spheroidal_width):
        pass

    # Wrapper to C function (override for each class inheriting from this)
    def _cwrap_degridding(
        w_step,
        shift,
        cell_size,
        kernel_size,
        subgrid_size,
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
        aterms_offsets_nr_timeslots,
        spheroidal,
        spheroidal_height,
        spheroidal_width):
        pass

    def _cwrap_calibrate(
        self,
        w_step,
        shift,
        cell_size,
        kernel_size,
        subgrid_size,
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
        aterms_offsets_nr_timeslots,
        spheroidal,
        spheroidal_height,
        spheroidal_width):
        self.lib.Proxy_calibrate(
            self.obj,
            ctypes.c_float(w_step),
            shift.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_float(cell_size),
            ctypes.c_int(kernel_size),
            ctypes.c_int(subgrid_size),
            frequencies.ctypes.data_as(ctypes.c_void_p),
            nr_channels,
            visibilities.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(visibilities_nr_baselines),
            ctypes.c_int(visibilities_nr_timesteps),
            ctypes.c_int(visibilities_nr_channels),
            ctypes.c_int(visibilities_nr_correlations),
            uvw.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(uvw_nr_baselines),
            ctypes.c_int(uvw_nr_timesteps),
            ctypes.c_int(uvw_nr_coordinates),
            baselines.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(baselines_nr_baselines),
            ctypes.c_int(baselines_two),
            grid.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(grid_nr_correlations),
            ctypes.c_int(grid_height),
            ctypes.c_int(grid_width),
            aterms.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(aterms_nr_timeslots),
            ctypes.c_int(aterms_nr_stations),
            ctypes.c_int(aterms_aterm_height),
            ctypes.c_int(aterms_aterm_width),
            ctypes.c_int(aterms_nr_correlations),
            aterms_offsets.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(aterms_offsets_nr_timeslots),
            spheroidal.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(spheroidal_height),
            ctypes.c_int(spheroidal_width))


    # Wrapper to C function (override for each class inheriting from this)
    def _cwrap_transform(
        self,
        direction,
        grid):
        pass

    def get_grid(
        self,
        nr_correlations,
        grid_size):

        # Get pointer to grid data
        lib.Proxy_get_grid.restype = ctypes.c_voidp
        ptr = lib.Proxy_get_grid(
            self.obj,
            ctypes.c_int(nr_correlations),
            ctypes.c_int(grid_size))

        # Get float pointer to grid data
        ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))

        # Construct numpy array out of this pointer
        shape = (nr_correlations, grid_size, grid_size)
        length = np.prod(shape[:])*2
        grid = np.ctypeslib.as_array(ptr, shape=(length,)).view(np.complex64)
        grid = grid.reshape(shape)

        # Return grid
        return grid
