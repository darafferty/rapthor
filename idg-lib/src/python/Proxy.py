import os
import ctypes
import numpy as np
import idg

class Proxy(object):

    def __del__(self):
        """Destroy"""
        self.lib.Proxy_destroy.argtypes = [ ctypes.c_void_p ]
        self.lib.Proxy_destroy(self.obj)

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
        self.lib.Proxy_gridding.argtypes = [
            ctypes.c_void_p, # proxy
            ctypes.c_float,  # w_step
            ctypes.c_void_p, # shift
            ctypes.c_float,  # cell_size
            ctypes.c_int,    # kernel_size
            ctypes.c_int,    # subgrid_size
            ctypes.c_void_p, # frequencies
            ctypes.c_int,
            ctypes.c_void_p, # visibilities
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p, # uvw
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p, # baselines
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p, # grid
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p, # aterms
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p, # aterms_offsets
            ctypes.c_int,
            ctypes.c_void_p, # spheroidal
            ctypes.c_int,
            ctypes.c_int]
        self.lib.Proxy_gridding(
            ctypes.c_void_p(self.obj),
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
        self.lib.Proxy_degridding.argtypes = [
            ctypes.c_void_p, # proxy
            ctypes.c_float,  # w_step
            ctypes.c_void_p, # shift
            ctypes.c_float,  # cell_size
            ctypes.c_int,    # kernel_size
            ctypes.c_int,    # subgrid_size
            ctypes.c_void_p, # frequencies
            ctypes.c_int,
            ctypes.c_void_p, # visibilities
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p, # uvw
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p, # baselines
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p, # grid
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p, # aterms
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p, # aterms_offsets
            ctypes.c_int,
            ctypes.c_void_p, # spheroidal
            ctypes.c_int,
            ctypes.c_int]
        self.lib.Proxy_degridding(
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

    def calibrate_init(
        self,
        w_step,
        shift,
        cell_size,
        kernel_size,
        subgrid_size,
        frequencies,
        visibilities,
        weights,
        uvw,
        baselines,
        grid,
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
        :param spheroidal: numpy.ndarray(
                shape=(height, width),
                dtype = idg.spheroidaltype)
        """
        # extract dimensions
        nr_channels     = frequencies.shape[0]
        nr_baselines    = visibilities.shape[0]
        nr_timesteps    = visibilities.shape[1]
        nr_correlations = visibilities.shape[3]
        grid_height     = grid.shape[1]
        grid_width      = grid.shape[2]
        nr_timeslots    = aterms_offsets.shape[0] - 1

        # call C function to do the work

        self.lib.Proxy_calibrate_init.argtypes = [
            ctypes.c_void_p,             #Proxy* p,
            ctypes.c_float,              #float w_step,
            np.ctypeslib.ndpointer(
                dtype=np.float32,
                shape=(3,),
                flags='C_CONTIGUOUS'),   #float* shift,
            ctypes.c_float,              #const float cell_size,
            ctypes.c_uint,               #unsigned int kernel_size,
            ctypes.c_uint,               #unsigned int subgrid_size,
            ctypes.c_uint,               #unsigned int nr_channels,
            ctypes.c_uint,               #unsigned int nr_baselines,
            ctypes.c_uint,               #unsigned int nr_timesteps,
            ctypes.c_uint,               #unsigned int nr_timeslots,
            ctypes.c_uint,               #unsigned int nr_correlations,
            ctypes.c_uint,               #unsigned int grid_height,
            ctypes.c_uint,               #unsigned int grid_width,
            np.ctypeslib.ndpointer(
                dtype=np.float32,
                shape=(nr_channels,),
                flags='C_CONTIGUOUS'),   #float* frequencies,
            np.ctypeslib.ndpointer(
                dtype=np.complex64,
                shape=(nr_baselines, nr_timesteps, nr_channels, nr_correlations),
                flags='C_CONTIGUOUS'),   #std::complex<float>* visibilities,
            np.ctypeslib.ndpointer(
                dtype=np.float32,
                shape=(nr_baselines, nr_timesteps, nr_channels, nr_correlations),
                flags='C_CONTIGUOUS'),   #float* weights,
            np.ctypeslib.ndpointer(
                dtype = idg.uvwtype,
                shape=(nr_baselines, nr_timesteps),
                flags='C_CONTIGUOUS'),   #float* uvw,
            np.ctypeslib.ndpointer(
                dtype=idg.baselinetype,
                shape=(nr_baselines,),
                flags='C_CONTIGUOUS'),   #unsigned int* baselines,
            np.ctypeslib.ndpointer(
                dtype=idg.gridtype,
                shape=(nr_correlations, grid_height, grid_width),
                flags='C_CONTIGUOUS'),   #std::complex<float>* grid,
            np.ctypeslib.ndpointer(
                dtype=np.int32,
                ndim=1,
                shape=(nr_timeslots+1, ),
                flags='C_CONTIGUOUS'),    #int* aterms_offsets);
            np.ctypeslib.ndpointer(
                dtype=np.float32,
                ndim=2,
                shape=(subgrid_size, subgrid_size),
                flags='C_CONTIGUOUS')    #float* spheroidal);
            ]

        self.lib.Proxy_calibrate_init(
            self.obj,
            w_step,
            shift,
            cell_size,
            kernel_size,
            subgrid_size,
            nr_channels,
            nr_baselines,
            nr_timesteps,
            nr_timeslots,
            nr_correlations,
            grid_height,
            grid_width,
            frequencies,
            visibilities,
            weights,
            uvw,
            baselines,
            grid,
            aterms_offsets,
            spheroidal)

    def calibrate_update(self, antenna_nr, aterms, aterm_derivatives, hessian, gradient, residual):

        nr_timeslots = aterms.shape[0]
        nr_antennas = aterms.shape[1]
        subgrid_size = aterms.shape[2]
        nr_terms = gradient.shape[1]
        nr_correlations = 4

        self.lib.Proxy_calibrate_update.argtypes = [
            ctypes.c_void_p,             #Proxy* p,
            ctypes.c_uint,               #unsigned int antenna_nr
            ctypes.c_uint,               #unsigned int subgrid_size
            ctypes.c_uint,               #unsigned int nr_antennas
            ctypes.c_uint,               #unsigned int nr_timeslots
            ctypes.c_uint,               #unsigned int nr_terms
            np.ctypeslib.ndpointer(
                dtype=np.complex64,
                shape=(nr_timeslots, nr_antennas, subgrid_size, subgrid_size, nr_correlations),
                flags='C_CONTIGUOUS'),   #std::complex<float>* aterms
            np.ctypeslib.ndpointer(
                dtype=np.complex64,
                shape=(nr_timeslots, nr_terms, subgrid_size, subgrid_size, nr_correlations),
                flags='C_CONTIGUOUS'),   #std::complex<float>* aterm_derivatives
            np.ctypeslib.ndpointer(
                dtype=np.float64,
                shape=(nr_timeslots, nr_terms, nr_terms),
                flags='C_CONTIGUOUS'),   #double* hessian
            np.ctypeslib.ndpointer(
                dtype=np.float64,
                shape=(nr_timeslots, nr_terms),
                flags='C_CONTIGUOUS'),   #double* gradient
            np.ctypeslib.ndpointer(
                dtype=np.float64,
                shape=(1, ),
                flags='C_CONTIGUOUS'),   #double* residual
            ]

        self.lib.Proxy_calibrate_update(
            self.obj,
            antenna_nr,
            subgrid_size,
            nr_antennas,
            nr_timeslots,
            nr_terms,
            aterms,
            aterm_derivatives,
            hessian,
            gradient,
            residual)

    def calibrate_init_hessian_vector_product(self):

        self.lib.Proxy_calibrate_init_hessian_vector_product.argtypes = [
            ctypes.c_void_p,             #Proxy* p,
            ]

        self.lib.Proxy_calibrate_init_hessian_vector_product(
            self.obj)

    def calibrate_hessian_vector_product1(self, antenna_nr, aterms, aterm_derivatives, parameter_vector):

        nr_timeslots = aterms.shape[0]
        nr_antennas = aterms.shape[1]
        subgrid_size = aterms.shape[2]
        nr_terms = parameter_vector.shape[1]
        nr_correlations = 4

        self.lib.Proxy_calibrate_hessian_vector_product1.argtypes = [
            ctypes.c_void_p,             #Proxy* p,
            ctypes.c_uint,               #unsigned int antenna_nr
            ctypes.c_uint,               #unsigned int subgrid_size
            ctypes.c_uint,               #unsigned int nr_antennas
            ctypes.c_uint,               #unsigned int nr_timeslots
            ctypes.c_uint,               #unsigned int nr_terms
            np.ctypeslib.ndpointer(
                dtype=np.complex64,
                shape=(nr_timeslots, nr_antennas, subgrid_size, subgrid_size, nr_correlations),
                flags='C_CONTIGUOUS'),   #std::complex<float>* aterms
            np.ctypeslib.ndpointer(
                dtype=np.complex64,
                shape=(nr_timeslots, nr_terms, subgrid_size, subgrid_size, nr_correlations),
                flags='C_CONTIGUOUS'),   #std::complex<float>* aterm_derivatives
            np.ctypeslib.ndpointer(
                dtype=np.float32,
                shape=(nr_timeslots, nr_terms),
                flags='C_CONTIGUOUS'),   #std::complex<float>* parameter_vector
            ]

        self.lib.Proxy_calibrate_hessian_vector_product1(
            self.obj,
            antenna_nr,
            subgrid_size,
            nr_antennas,
            nr_timeslots,
            nr_terms,
            aterms,
            aterm_derivatives,
            parameter_vector)

    def calibrate_hessian_vector_product2(self, antenna_nr, aterms, aterm_derivatives, parameter_vector):

        nr_timeslots = aterms.shape[0]
        nr_antennas = aterms.shape[1]
        subgrid_size = aterms.shape[2]
        nr_terms = parameter_vector.shape[1]
        nr_correlations = 4

        self.lib.Proxy_calibrate_hessian_vector_product2.argtypes = [
            ctypes.c_void_p,             #Proxy* p,
            ctypes.c_uint,               #unsigned int antenna_nr
            ctypes.c_uint,               #unsigned int subgrid_size
            ctypes.c_uint,               #unsigned int nr_antennas
            ctypes.c_uint,               #unsigned int nr_timeslots
            ctypes.c_uint,               #unsigned int nr_terms
            np.ctypeslib.ndpointer(
                dtype=np.complex64,
                shape=(nr_timeslots, nr_antennas, subgrid_size, subgrid_size, nr_correlations),
                flags='C_CONTIGUOUS'),   #std::complex<float>* aterms
            np.ctypeslib.ndpointer(
                dtype=np.complex64,
                shape=(nr_timeslots, nr_terms, subgrid_size, subgrid_size, nr_correlations),
                flags='C_CONTIGUOUS'),   #std::complex<float>* aterm_derivatives
            np.ctypeslib.ndpointer(
                dtype=np.float32,
                shape=(nr_timeslots, nr_terms),
                flags='C_CONTIGUOUS'),   #float* parameter_vector
            ]
        self.lib.Proxy_calibrate_hessian_vector_product2(
            self.obj,
            antenna_nr,
            subgrid_size,
            nr_antennas,
            nr_timeslots,
            nr_terms,
            aterms,
            aterm_derivatives,
            parameter_vector)


    def calibrate_finish(self):

        self.lib.Proxy_calibrate_finish.argtypes = [
            ctypes.c_void_p,             #Proxy* p,
            ]

        self.lib.Proxy_calibrate_finish(
            self.obj)

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
        self.lib.Proxy_transform.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int]
        self.lib.Proxy_transform(
            self.obj,
            ctypes.c_int(direction),
            grid.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nr_correlations),
            ctypes.c_int(height),
            ctypes.c_int(width))

    def allocate_grid(
        self,
        nr_correlations,
        grid_size):

        # Get pointer to grid data
        self.lib.Proxy_allocate_grid.restype = ctypes.c_void_p
        self.lib.Proxy_allocate_grid.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int]
        ptr = self.lib.Proxy_allocate_grid(
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
