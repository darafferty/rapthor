# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

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
        kernel_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        aterms,
        aterms_offsets,
        taper):
        """
        Grid visibilities onto grid.

        :param kernel_size: int
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
        :param aterms: numpy.ndarray(
                shape=(nr_timeslots, nr_stations, height, width, nr_correlations),
                dtype = idg.atermtype)
        :param aterms_offsets: numpy.ndarray(
                shape=(nr_timeslots+1),
                dtype = idg.atermoffsettype)
        :param taper: numpy.ndarray(
                shape=(height, width),
                dtype = idg.tapertype)
        """
        # extract dimensions
        nr_channels = frequencies.shape[0]
        nr_baselines    = visibilities.shape[0]
        nr_timesteps    = visibilities.shape[1]
        nr_correlations = 4
        nr_timeslots       = aterms.shape[0]
        nr_stations        = aterms.shape[1]
        subgrid_size       = aterms.shape[2]

        #Set C function signature
        self.lib.Proxy_gridding.argtypes = [
            ctypes.c_void_p, # proxy
            ctypes.c_int,    # kernel_size
            ctypes.c_int,    # subgrid_size
            ctypes.c_int,    # nr_channels
            ctypes.c_int,    # nr_baselines
            ctypes.c_int,    # nr_timesteps,
            ctypes.c_int,    # nr_correlations,
            ctypes.c_int,    # nr_timeslots
            ctypes.c_int,    # int nr_stations
            np.ctypeslib.ndpointer(
                dtype=idg.frequenciestype,
                shape=(nr_channels,),
                flags='C_CONTIGUOUS'),   # frequencies
            np.ctypeslib.ndpointer(
                dtype=idg.visibilitiestype,
                shape=(nr_baselines, nr_timesteps, nr_channels, nr_correlations),
                flags='C_CONTIGUOUS'), # visibilities
            np.ctypeslib.ndpointer(
                dtype=idg.uvwtype,
                shape=(nr_baselines, nr_timesteps),
                flags='C_CONTIGUOUS'), # uvw
            np.ctypeslib.ndpointer(
                dtype=idg.baselinetype,
                shape=(nr_baselines,),
                flags='C_CONTIGUOUS'),# baselines
            np.ctypeslib.ndpointer(
                dtype=idg.atermtype,
                shape=(nr_timeslots, nr_stations, subgrid_size, subgrid_size, nr_correlations),
                flags='C_CONTIGUOUS'), # aterms
            np.ctypeslib.ndpointer(
                dtype=idg.atermoffsettype,
                shape=(nr_timeslots+1, ),
                flags='C_CONTIGUOUS'), # aterms_offsets
            np.ctypeslib.ndpointer(
                dtype=idg.tapertype,
                shape=(subgrid_size, subgrid_size),
                flags='C_CONTIGUOUS')] # taper
        # call C function to do the work
        self.lib.Proxy_gridding(
            self.obj,
            kernel_size,
            subgrid_size,
            nr_channels,
            nr_baselines,
            nr_timesteps,
            nr_correlations,
            nr_timeslots,
            nr_stations,
            frequencies,
            visibilities,
            uvw,
            baselines,
            aterms,
            aterms_offsets,
            taper)

    def degridding(
        self,
        kernel_size,
        frequencies,
        visibilities,
        uvw,
        baselines,
        aterms,
        aterms_offsets,
        taper):
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
        :param aterms: numpy.ndarray(
                shape=(nr_timeslots, nr_stations, height, width, nr_correlations),
                dtype = idg.atermtype)
        :param aterms_offsets: numpy.ndarray(
                shape=(nr_timeslots+1),
                dtype = idg.atermoffsettype)
        :param taper: numpy.ndarray(
                shape=(height, width),
                dtype = idg.tapertype)
        """
        # extract dimensions
        nr_channels = frequencies.shape[0]
        nr_baselines    = visibilities.shape[0]
        nr_timesteps    = visibilities.shape[1]
        nr_correlations = 4
        nr_timeslots       = aterms.shape[0]
        nr_stations        = aterms.shape[1]
        subgrid_size       = aterms.shape[2]

        # Set C function signature
        self.lib.Proxy_degridding.argtypes = [
            ctypes.c_void_p, # proxy
            ctypes.c_int,    # kernel_size
            ctypes.c_int,    # subgrid_size
            ctypes.c_int,    # nr_channels
            ctypes.c_int,    # nr_baselines
            ctypes.c_int,    # nr_timesteps,
            ctypes.c_int,    # nr_correlations,
            ctypes.c_int,    # nr_timeslots
            ctypes.c_int,    # int nr_stations
            np.ctypeslib.ndpointer(
                dtype=idg.frequenciestype,
                shape=(nr_channels,),
                flags='C_CONTIGUOUS'),   # frequencies
            np.ctypeslib.ndpointer(
                dtype=idg.visibilitiestype,
                shape=(nr_baselines, nr_timesteps, nr_channels, nr_correlations),
                flags='C_CONTIGUOUS'), # visibilities
            np.ctypeslib.ndpointer(
                dtype=idg.uvwtype,
                shape=(nr_baselines, nr_timesteps),
                flags='C_CONTIGUOUS'), # uvw
            np.ctypeslib.ndpointer(
                dtype=idg.baselinetype,
                shape=(nr_baselines,),
                flags='C_CONTIGUOUS'),# baselines
            np.ctypeslib.ndpointer(
                dtype=idg.atermtype,
                shape=(nr_timeslots, nr_stations, subgrid_size, subgrid_size, nr_correlations),
                flags='C_CONTIGUOUS'), # aterms
            np.ctypeslib.ndpointer(
                dtype=idg.atermoffsettype,
                shape=(nr_timeslots+1, ),
                flags='C_CONTIGUOUS'), # aterms_offsets
            np.ctypeslib.ndpointer(
                dtype=idg.tapertype,
                shape=(subgrid_size, subgrid_size),
                flags='C_CONTIGUOUS')] # taper
        # call C function to do the work
        self.lib.Proxy_degridding(
            self.obj,
            kernel_size,
            subgrid_size,
            nr_channels,
            nr_baselines,
            nr_timesteps,
            nr_correlations,
            nr_timeslots,
            nr_stations,
            frequencies,
            visibilities,
            uvw,
            baselines,
            aterms,
            aterms_offsets,
            taper)

    def init_cache(self, subgrid_size, cell_size, w_step, shift):
        self.lib.Proxy_init_cache.argtypes = [
            ctypes.c_void_p,             #Proxy* p,
            ctypes.c_uint,               #unsigned int subgrid_size,
            ctypes.c_float,              #const float cell_size,
            ctypes.c_float,              #float w_step,
            np.ctypeslib.ndpointer(
                dtype=np.float32,
                shape=(3,),
                flags='C_CONTIGUOUS')]   #float* shift,
        self.lib.Proxy_init_cache(
            self.obj,
            subgrid_size,
            cell_size,
            w_step,
            shift)

    def calibrate_init(
        self,
        kernel_size,
        frequencies,
        visibilities,
        weights,
        uvw,
        baselines,
        aterms_offsets,
        taper):
        """
        Calibrate

        :param frequencies: numpy.ndarray(
                shape = (nr_channels,)
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
        :param taper: numpy.ndarray(
                shape=(height, width),
                dtype = idg.tapertype)
        """
        # extract dimensions
        subgrid_size    = taper.shape[0]
        nr_channels     = frequencies.shape[0]
        nr_baselines    = visibilities.shape[0]
        nr_timesteps    = visibilities.shape[1]
        nr_correlations = visibilities.shape[3]
        nr_timeslots    = aterms_offsets.shape[0] - 1

        # call C function to do the work

        self.lib.Proxy_calibrate_init.argtypes = [
            ctypes.c_void_p,             #Proxy* p,
            ctypes.c_uint,               #unsigned int kernel_size,
            ctypes.c_uint,               #unsigned int subgrid_size,
            ctypes.c_uint,               #unsigned int nr_channels,
            ctypes.c_uint,               #unsigned int nr_baselines,
            ctypes.c_uint,               #unsigned int nr_timesteps,
            ctypes.c_uint,               #unsigned int nr_timeslots,
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
                dtype=np.int32,
                ndim=1,
                shape=(nr_timeslots+1, ),
                flags='C_CONTIGUOUS'),    # aterms_offsets
            np.ctypeslib.ndpointer(
                dtype=np.float32,
                ndim=2,
                shape=(subgrid_size, subgrid_size),
                flags='C_CONTIGUOUS')    # taper
            ]

        self.lib.Proxy_calibrate_init(
            self.obj,
            kernel_size,
            subgrid_size,
            nr_channels,
            nr_baselines,
            nr_timesteps,
            nr_timeslots,
            frequencies,
            visibilities,
            weights,
            uvw,
            baselines,
            aterms_offsets,
            taper)

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
        direction):
        """
        Transform Fourier Domain<->Image Domain.

        :param direction: idg.FourierDomainToImageDomain or idg.ImageDomainToFourierDomain
        """

        # call C function to do the work
        self.lib.Proxy_transform.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int]
        self.lib.Proxy_transform(
            ctypes.c_void_p(self.obj),
            ctypes.c_int(direction))

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
            ctypes.c_void_p(self.obj),
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

    def set_grid(
        self,
        grid):

        """
        Set grid to use in proxy

        :param grid: numpy.ndarray(
                shape=(nr_correlations, height, width),
                dtype = idg.gridtype)
        """

        # Get dimensions
        shape = grid.shape
        nr_w_layers = 1
        nr_correlations = shape[0]
        grid_size = shape[1]
        height = grid_size
        width = grid_size

        # Set argument types
        self.lib.Proxy_set_grid.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int]

        # Call the C function
        self.lib.Proxy_set_grid(
            ctypes.c_void_p(self.obj),
            grid.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nr_w_layers),
            ctypes.c_int(nr_correlations),
            ctypes.c_int(height),
            ctypes.c_int(width))

    def get_final_grid(
        self,
        grid):

        """
        Retrieve grid from proxy

        :param grid: numpy.ndarray(
                shape=(nr_correlations, height, width),
                dtype = idg.gridtype)
        """

        # Get dimensions
        shape = grid.shape
        nr_w_layers = 1
        nr_correlations = shape[0]
        grid_size = shape[1]
        height = grid_size
        width = grid_size

        # Set argument types
        self.lib.Proxy_get_final_grid.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int]

        # Call the C function
        self.lib.Proxy_get_final_grid(
            ctypes.c_void_p(self.obj),
            grid.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nr_w_layers),
            ctypes.c_int(nr_correlations),
            ctypes.c_int(height),
            ctypes.c_int(width))