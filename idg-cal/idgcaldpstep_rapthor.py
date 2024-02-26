# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import everybeam as eb
import numpy as np
import idg
from idg.h5parmwriter import H5ParmWriter
from idg.basisfunctions import LagrangePolynomial
from idg.idgcaldpstepbase import IDGCalDPStepBase
from idg.idgcalutils import (
    apply_beam,
    expand_basis_functions,
    next_composite,
    idgwindow,
    get_aterm_offsets,
    init_h5parm_solution_table,
)
import astropy.io.fits as fits
import scipy.linalg
import time
import logging


class IDGCalDPStepRapthor(IDGCalDPStepBase):
    def _update_info(self, dpinfo):
        super()._update_info(dpinfo)
        self.ms_name = dpinfo.ms_name
        if self.usebeammodel:
            self.telescope = eb.load_telescope(
                self.ms_name, beam_normalisation_mode=self.beamnormalisationmode
            )
        if self.antenna_constraint:
            self.antenna_constraint = [
                [self.info.antenna_names.index(antenna) for antenna in antenna_group]
                for antenna_group in self.antenna_constraint
            ]
            self.A_antenna_constraint = np.eye(self.info.n_antenna)
            for antenna_group in self.antenna_constraint:
                self.A_antenna_constraint[
                    np.ix_(antenna_group, antenna_group)
                ] = 1.0 / len(antenna_group)

    def read_parset(self, parset, prefix):
        """
        Read relevant information from a given parset

        Parameters
        ----------
        parset : dp3.ParameterSet
            ParameterSet object provided by DP3
        prefix : str
            Prefix to be used when reading the parset.
        """

        self.proxytype = parset.get_string(prefix + "proxytype", "CPU")

        self.write_model_data = parset.get_bool(prefix + "writemodeldata", False)
        self.usebeammodel = parset.get_bool(prefix + "usebeammodel", False)
        self.beammode = parset.get_string(prefix + "beammode", "default")
        self.beamnormalisationmode = eb.parse_beam_normalisation_mode(
            parset.get_string(prefix + "beamnormalisationmode", "full")
        )

        solint = parset.get_int(prefix + "solint", 0)
        if solint:
            self.ampl_interval = solint
            self.phase_interval = solint
        else:
            self.ampl_interval = parset.get_int(prefix + "solintamplitude", 0)
            self.phase_interval = parset.get_int(prefix + "solintphase", 0)

        # solintamplitude should be divisible by solintphase, check and correct if that's not the case
        remainder = self.ampl_interval % self.phase_interval
        if remainder != 0:
            logging.warning(
                f"Specified amplitude solution interval {self.ampl_interval} is not an integer multiple of the phase solution interval {self.phase_interval}. Amplitude soluton interval will be modified to {self.ampl_interval + remainder}"
            )
            self.ampl_interval += remainder

        self.imagename = parset.get_string(prefix + "modelimage")
        self.padding = parset.get_float(prefix + "padding", 1.2)
        self.nr_correlations = parset.get_int(prefix + "nrcorrelations", 4)
        self.subgrid_size = parset.get_int(prefix + "subgridsize", 32)

        self.taper_support = parset.get_int(prefix + "tapersupport", 7)
        wterm_support = parset.get_int(prefix + "wtermsupport", 5)
        aterm_support = parset.get_int(prefix + "atermsupport", 5)
        self.kernel_size = self.taper_support + wterm_support + aterm_support

        # get polynomial degrees for amplitude/phase
        ampl_order = parset.get_int(prefix + "polynomialdegamplitude", 2)
        phase_order = parset.get_int(prefix + "polynomialdegphase", 1)

        # Solver related
        # Factor between 0 and 1 with which to update solution between iterations
        self.solver_update_gain = parset.get_float(prefix + "solverupdategain", 0.5)
        # Tolerance pseudo inverse
        self.pinv_tol = parset.get_double(prefix + "tolerancepinv", 1e-9)
        # Maximum number of iterations
        self.max_iter = parset.get_int(prefix + "maxiter", 1)

        # H5Parm output file related
        self.h5parm_fname = parset.get_string(prefix + "h5parm", "idgcal.h5")
        self.h5parm_solsetname = parset.get_string(
            prefix + "h5parmsolset", "coefficients000"
        )
        self.h5parm_overwrite = parset.get_bool(prefix + "h5parmoverwrite", True)

        self.w_step = parset.get_float(prefix + "wstep", 400.0)

        self.nr_channels_per_block = parset.get_int(prefix + "nr_channels_per_block", 0)
        self.apply_phase_constraint = parset.get_bool(
            prefix + "apply_phase_constraint", False
        )

        if prefix + "antennaconstraint" in parset:
            self.antenna_constraint = [
                [str(antenna) for antenna in constraint]
                for constraint in parset[prefix + "antennaconstraint"]
            ]
        else:
            self.antenna_constraint = None

        self.shift = np.array((0.0, 0.0), dtype=np.float32)

        # Number of phase updates per amplitude interval
        self.nr_phase_updates = self.ampl_interval // self.phase_interval

        # Initialize amplitude and phase polynomial
        self.ampl_poly = LagrangePolynomial(order=ampl_order)
        self.phase_poly = LagrangePolynomial(order=phase_order)

        self.nr_parameters0 = 4 * self.ampl_poly.nr_coeffs + self.phase_poly.nr_coeffs
        self.nr_parameters = (
            4 * self.ampl_poly.nr_coeffs
            + self.phase_poly.nr_coeffs * self.nr_phase_updates
        )

    def initialize(self):
        self.is_initialized = True

        # Counter for the number of calls to process_buffers
        self.count_process_buffer_calls = 0

        # Extract the time info and cast into a time array
        tstart = self.info.start_time
        # Time array should match "amplitude time blocks"
        nsteps = (self.info.n_times // self.ampl_interval) * self.ampl_interval
        dt = self.info.time_interval
        time_array = np.linspace(
            tstart, tstart + dt * nsteps, num=nsteps, endpoint=False
        )
        # Get time centroids per amplitude/phase solution interval
        self.time_array_ampl = (
            time_array[:: self.ampl_interval] + (self.ampl_interval - 1) * dt / 2.0
        )
        self.time_array_phase = (
            time_array[:: self.phase_interval] + (self.phase_interval - 1) * dt / 2.0
        )

        self.nr_stations = self.info.n_antenna
        self.nr_baselines = (self.nr_stations * (self.nr_stations - 1)) // 2
        self.frequencies = np.array(self.info.channel_frequencies, dtype=np.float32)
        self.nr_channels = len(self.frequencies)

        if self.nr_channels_per_block == 0:
            self.nr_channels_per_block = self.nr_channels
        self.nr_channel_blocks = self.nr_channels // self.nr_channels_per_block
        assert self.nr_channel_blocks * self.nr_channels_per_block == self.nr_channels

        self.frequencies = self.frequencies.reshape(
            (self.nr_channel_blocks, self.nr_channels_per_block)
        )
        self.freq_array = np.mean(self.frequencies, axis=-1)

        self.baselines = np.zeros(shape=(self.nr_baselines, 2), dtype=np.int32)

        station1 = np.array(self.info.first_antenna_indices)
        station2 = np.array(self.info.second_antenna_indices)
        self.auto_corr_mask = station1 != station2
        self.baselines[:, 0] = station1[self.auto_corr_mask]
        self.baselines[:, 1] = station2[self.auto_corr_mask]

        # Axes data
        axes_labels = ["freq", "ant", "time", "dir"]
        axes_data_amplitude = dict(
            zip(
                axes_labels,
                (
                    self.nr_channel_blocks,
                    self.nr_stations,
                    self.time_array_ampl.size,
                    self.ampl_poly.nr_coeffs,
                ),
            )
        )
        axes_data_phase = dict(
            zip(
                axes_labels,
                (
                    self.nr_channel_blocks,
                    self.nr_stations,
                    self.time_array_phase.size,
                    self.phase_poly.nr_coeffs,
                ),
            )
        )

        # Initialize h5parm file
        self.h5writer = H5ParmWriter(
            self.h5parm_fname,
            solution_set_name=self.h5parm_solsetname,
            overwrite=self.h5parm_overwrite,
        )

        # Add antenna/station info
        self.h5writer.add_antennas(self.info.antenna_names, self.info.antenna_positions)

        if self.proxytype.lower() == "gpu":
            self.proxy = idg.HybridCUDA.GenericOptimized()
        else:
            self.proxy = idg.CPU.Optimized()

        # read image dimensions from fits header
        h = fits.getheader(self.imagename)
        N0 = h["NAXIS1"]
        self.cell_size = np.deg2rad(abs(h["CDELT1"]))

        # Pointing of image
        # TODO This should be checked against the pointing in the MS
        self.ra = np.deg2rad(h["CRVAL1"])
        self.dec = np.deg2rad(h["CRVAL2"])

        # compute padded image size
        N = next_composite(int(N0 * self.padding))
        self.grid_size = N
        self.image_size = N * self.cell_size

        # Initialize solution tables for amplitude and phase coefficients
        # TODO: maybe pass parset as string to HISTORY?
        init_h5parm_solution_table(
            self.h5writer,
            "amplitude1",
            axes_data_amplitude,
            self.info.antenna_names,
            self.time_array_ampl,
            self.freq_array,
            self.image_size,
            self.subgrid_size,
        )
        init_h5parm_solution_table(
            self.h5writer,
            "amplitude2",
            axes_data_amplitude,
            self.info.antenna_names,
            self.time_array_ampl,
            self.freq_array,
            self.image_size,
            self.subgrid_size,
        )
        init_h5parm_solution_table(
            self.h5writer,
            "slowphase1",
            axes_data_amplitude,
            self.info.antenna_names,
            self.time_array_ampl,
            self.freq_array,
            self.image_size,
            self.subgrid_size,
        )
        init_h5parm_solution_table(
            self.h5writer,
            "slowphase2",
            axes_data_amplitude,
            self.info.antenna_names,
            self.time_array_ampl,
            self.freq_array,
            self.image_size,
            self.subgrid_size,
        )
        init_h5parm_solution_table(
            self.h5writer,
            "phase",
            axes_data_phase,
            self.info.antenna_names,
            self.time_array_phase,
            self.freq_array,
            self.image_size,
            self.subgrid_size,
        )

        # Initialize empty grid
        self.grid = np.zeros(
            shape=(self.nr_correlations, self.grid_size, self.grid_size),
            dtype=idg.gridtype,
        )

        # Initialize taper
        taper = idgwindow(self.subgrid_size, self.taper_support, self.padding)
        self.taper2 = np.outer(taper, taper).astype(np.float32)

        taper_ = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(taper)))
        taper_grid = np.zeros(self.grid_size, dtype=np.complex128)
        taper_grid[
            (self.grid_size - self.subgrid_size)
            // 2 : (self.grid_size + self.subgrid_size)
            // 2
        ] = taper_ * np.exp(
            -1j * np.linspace(-np.pi / 2, np.pi / 2, self.subgrid_size, endpoint=False)
        )
        taper_grid = (
            np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(taper_grid))).real
            * self.grid_size
            / self.subgrid_size
        )
        taper_grid0 = taper_grid[(N - N0) // 2 : (N + N0) // 2]

        # read image data, assume Stokes I
        d = fits.getdata(self.imagename)

        assert not np.any(np.isnan(d))
        self.grid[0, (N - N0) // 2 : (N + N0) // 2, (N - N0) // 2 : (N + N0) // 2] = d[
            0, 0, :, :
        ] / np.outer(taper_grid0, taper_grid0)
        self.grid[3, (N - N0) // 2 : (N + N0) // 2, (N - N0) // 2 : (N + N0) // 2] = d[
            0, 0, :, :
        ] / np.outer(taper_grid0, taper_grid0)

        self.proxy.set_grid(self.grid)
        self.proxy.transform(idg.ImageDomainToFourierDomain)

        self.proxy.init_cache(
            self.subgrid_size, self.cell_size, self.w_step, self.shift
        )

        self.aterm_offsets = get_aterm_offsets(
            self.nr_phase_updates, self.ampl_interval
        )

        self.Bampl1, self.Tampl = expand_basis_functions(
            self.ampl_poly, self.subgrid_size, self.image_size, 1.0, 0.0
        )
        self.Bampl2, self.Tampl = expand_basis_functions(
            self.ampl_poly, self.subgrid_size, self.image_size, 0.0, 1.0
        )
        self.Bslowphase1, self.Tampl = expand_basis_functions(
            self.ampl_poly, self.subgrid_size, self.image_size, 1.0, 0.0
        )
        self.Bslowphase2, self.Tampl = expand_basis_functions(
            self.ampl_poly, self.subgrid_size, self.image_size, 0.0, 1.0
        )
        self.Bphase, self.Tphase = expand_basis_functions(
            self.phase_poly, self.subgrid_size, self.image_size
        )

        gs = eb.GridSettings()
        gs.width = gs.height = self.subgrid_size
        gs.ra = self.ra
        gs.dec = self.dec
        gs.dl = -self.image_size / self.subgrid_size
        gs.dm = -self.image_size / self.subgrid_size
        gs.l_shift = -0.5 * gs.dl
        gs.m_shift = 0.5 * gs.dm
        self.gs = gs

        freqs = np.mean(self.frequencies, axis=-1)
        A = np.array((np.ones(self.nr_channel_blocks), 1 / freqs)).T
        if self.apply_phase_constraint:
            self.constraint_matrix = A @ np.linalg.inv(A.T @ A) @ A.T

    def process_buffers(self):
        """
        Processing the buffers. This is the central method within any class that
        derives from dp3.Step
        """

        if not self.is_initialized:
            self.initialize()

        # Concatenate accumulated data and display just the shapes
        visibilities = self._extract_buffer("visibilities")
        flags = self._extract_buffer("flags")
        weights = self._extract_buffer("weights")
        uvw_ = self._extract_buffer("uvw", apply_autocorr_mask=False)
        uvw = np.zeros(
            shape=(self.nr_baselines, self.ampl_interval, 3), dtype=np.float32
        )

        uvw[..., 0] = uvw_[self.auto_corr_mask, :, 0]
        uvw[..., 1] = -uvw_[self.auto_corr_mask, :, 1]
        uvw[..., 2] = -uvw_[self.auto_corr_mask, :, 2]

        times = np.array([dpbuffer.get_time() for dpbuffer in self.dpbuffers])

        # Flag NaNs
        flags[np.isnan(visibilities)] = True

        # Set weights of flagged visibilities to zero
        weights *= ~flags

        # Even with weight=0, NaNs still propagate, so set NaN visiblities to zero
        visibilities[np.isnan(visibilities)] = 0.0

        if self.usebeammodel:
            # TODO Pass self.beammode here once the gridded_response function
            # from the EveryBeam python interface supports that
            aterms_beam = self.telescope.gridded_response(
                self.gs, np.mean(times), np.mean(self.frequencies)
            )
        else:
            aterms_beam = None

        self.proxy.calibrate_init(
            self.kernel_size,
            self.frequencies,
            visibilities,
            weights,
            uvw,
            self.baselines,
            self.aterm_offsets,
            self.taper2,
        )

        # Initialize coefficients, both for amplitude and phase
        # The amplitude coefficients are initialized with ones for the constant in the polynomial expansion (X0)
        # and zeros otherwise (X1). The parameters for the phases are initialized with zeros (X2).
        X0 = np.ones((self.nr_channel_blocks, self.nr_stations, 1))
        X1 = np.zeros(
            (self.nr_channel_blocks, self.nr_stations, self.ampl_poly.nr_coeffs - 1)
        )
        X2 = np.zeros(
            (
                self.nr_channel_blocks,
                self.nr_stations,
                self.nr_parameters - 4 * self.ampl_poly.nr_coeffs,
            )
        )

        parameters = np.concatenate(
            (X0, X1, X0, X1, 0 * X0, X1, 0 * X0, X1, X2), axis=-1
        )
        # Map parameters to orthonormal basis
        parameters = transform_parameters(
            np.linalg.inv(self.Tampl),
            np.linalg.inv(self.Tphase),
            parameters,
            self.ampl_poly.nr_coeffs,
            self.phase_poly.nr_coeffs,
            self.nr_stations,
            self.nr_phase_updates,
        )

        # Compute amplitude and phase aterms from parameters
        # The first :self.ampl_poly.nr_coeffs of the last axis of parameters
        # contain the coefficients for the amplitude
        # the remaining entries are the coefficients for the phase
        aterm_ampl1 = np.tensordot(
            parameters[:, :, : self.ampl_poly.nr_coeffs], self.Bampl1, axes=((2,), (0,))
        )
        aterm_ampl2 = np.tensordot(
            parameters[:, :, self.ampl_poly.nr_coeffs : 2 * self.ampl_poly.nr_coeffs],
            self.Bampl2,
            axes=((2,), (0,)),
        )
        aterm_slowphase1 = np.exp(
            1j
            * np.tensordot(
                parameters[
                    :, :, 2 * self.ampl_poly.nr_coeffs : 3 * self.ampl_poly.nr_coeffs
                ],
                self.Bslowphase1,
                axes=((2,), (0,)),
            )
        )
        aterm_slowphase2 = np.exp(
            1j
            * np.tensordot(
                parameters[
                    :, :, 3 * self.ampl_poly.nr_coeffs : 4 * self.ampl_poly.nr_coeffs
                ],
                self.Bslowphase2,
                axes=((2,), (0,)),
            )
        )

        aterm_slow = aterm_ampl1 * aterm_slowphase1 + aterm_ampl2 * aterm_slowphase2

        aterm_phase = np.exp(
            1j
            * np.tensordot(
                parameters[:, :, 4 * self.ampl_poly.nr_coeffs :].reshape(
                    (
                        self.nr_channel_blocks,
                        self.nr_stations,
                        self.nr_phase_updates,
                        self.phase_poly.nr_coeffs,
                    )
                ),
                self.Bphase,
                axes=((3,), (0,)),
            )
        )

        # Transpose swaps the station and phase_update axes
        aterms = (
            aterm_phase.transpose((0, 2, 1, 3, 4, 5))
            * aterm_slow[:, np.newaxis, :, :, :, :]
        )

        aterms = apply_beam(aterms_beam, aterms)

        aterms = np.ascontiguousarray(aterms.astype(idg.idgtypes.atermtype))

        nr_iterations = 0
        converged = False
        previous_residual = 0.0

        max_dx = 0.0

        timer = -time.time()
        timer0 = 0
        timer1 = 0

        while True:
            nr_iterations += 1
            logging.debug(f"iteration nr {nr_iterations} ")

            max_dx = 0.0
            norm_dx = 0.0
            residual_sum = 0.0
            for i in range(self.nr_stations):
                logging.debug(f"   Station {i}")
                timer1 -= time.time()

                # Predict visibilities for current solution
                hessian = np.zeros(
                    (
                        self.nr_channel_blocks,
                        self.nr_phase_updates,
                        self.nr_parameters0,
                        self.nr_parameters0,
                    ),
                    dtype=np.float64,
                )
                gradient = np.zeros(
                    (
                        self.nr_channel_blocks,
                        self.nr_phase_updates,
                        self.nr_parameters0,
                    ),
                    dtype=np.float64,
                )
                residual = np.zeros((self.nr_channel_blocks,), dtype=np.float64)

                aterm_ampl = self.__compute_amplitude(i, parameters)
                aterm_phase = self.__compute_phase(i, parameters)

                if aterms_beam is not None:
                    aterm_derivatives = compute_aterm_derivatives(
                        aterm_ampl,
                        aterm_phase,
                        aterms_beam[i],
                        self.Bampl1,
                        self.Bampl2,
                        self.Bslowphase1,
                        self.Bslowphase2,
                        self.Bphase,
                    )
                else:
                    aterm_derivatives = compute_aterm_derivatives(
                        aterm_ampl,
                        aterm_phase,
                        None,
                        self.Bampl1,
                        self.Bampl2,
                        self.Bslowphase1,
                        self.Bslowphase2,
                        self.Bphase,
                    )

                timer0 -= time.time()
                self.proxy.calibrate_update(
                    i, aterms, aterm_derivatives, hessian, gradient, residual
                )

                timer0 += time.time()
                residual_sum += residual[0]

                gradient = np.concatenate(
                    (
                        np.sum(gradient[:, :, : 4 * self.ampl_poly.nr_coeffs], axis=1),
                        gradient[:, :, 4 * self.ampl_poly.nr_coeffs :].reshape(
                            (self.nr_channel_blocks, -1)
                        ),
                    ),
                    axis=1,
                )

                H00 = hessian[
                    :, :, : 4 * self.ampl_poly.nr_coeffs, : 4 * self.ampl_poly.nr_coeffs
                ].sum(axis=1)
                H01 = np.concatenate(
                    [
                        hessian[
                            :,
                            t,
                            : 4 * self.ampl_poly.nr_coeffs,
                            4 * self.ampl_poly.nr_coeffs :,
                        ]
                        for t in range(self.nr_phase_updates)
                    ],
                    axis=2,
                )
                H10 = np.concatenate(
                    [
                        hessian[
                            :,
                            t,
                            4 * self.ampl_poly.nr_coeffs :,
                            : 4 * self.ampl_poly.nr_coeffs,
                        ]
                        for t in range(self.nr_phase_updates)
                    ],
                    axis=1,
                )
                H11 = np.concatenate(
                    [
                        scipy.linalg.block_diag(
                            *[
                                hessian[
                                    sb,
                                    t,
                                    4 * self.ampl_poly.nr_coeffs :,
                                    4 * self.ampl_poly.nr_coeffs :,
                                ]
                                for t in range(self.nr_phase_updates)
                            ]
                        )[np.newaxis, :, :]
                        for sb in range(self.nr_channel_blocks)
                    ]
                )

                hessian = np.concatenate(
                    [
                        np.block([[H00[sb], H01[sb]], [H10[sb], H11[sb]]])[
                            np.newaxis, :, :
                        ]
                        for sb in range(self.nr_channel_blocks)
                    ]
                )

                # Per channel_group, apply the inverse of the Hessian to the gradient
                # s is the channel_group index, ij are the rows and columns of the Hessian
                try:
                    dx = np.einsum(
                        "sij,sj->si", np.linalg.pinv(hessian, self.pinv_tol), gradient
                    )
                except np.linalg.LinAlgError:
                    logging.warning("LinAlgError, could not update")
                    logging.warning(hessian)
                    timer1 += time.time()
                    continue

                if max_dx < np.amax(abs(dx)):
                    max_dx = np.amax(abs(dx))

                parameters[:, i] += self.solver_update_gain * dx

                # If requested, apply constraint to phase parameters
                if self.apply_phase_constraint:
                    parameters[:, i, 4 * self.ampl_poly.nr_coeffs :] = (
                        self.constraint_matrix
                        @ parameters[:, i, 4 * self.ampl_poly.nr_coeffs :]
                    )

                timer1 += time.time()

            # Apply antenna constraint
            if self.antenna_constraint:
                parameters = np.einsum(
                    "ij,kjl->kil", self.A_antenna_constraint, parameters
                )

            # Recompute aterms with updated parameters
            for i in range(self.nr_stations):

                aterm_ampl = self.__compute_amplitude(i, parameters)
                aterm_phase = self.__compute_phase(i, parameters)

                aterms_i = aterm_ampl * aterm_phase

                if aterms_beam is not None:
                    aterms_i = apply_beam(aterms_beam[i], aterms_i)

                aterms[:, :, i] = aterms_i

            dresidual = previous_residual - residual_sum
            fractional_dresidual = dresidual / residual_sum

            logging.debug(
                f"max_dx: {max_dx}, residual_sum: {residual_sum}, dresidual: {dresidual}, fractional_dresidual: {fractional_dresidual}"
            )

            previous_residual = residual_sum

            converged = (nr_iterations > 1) and (fractional_dresidual < 1e-5)
            # converged = (nr_iterations > 1) and (max_dx < 1e-2)

            if converged:
                logging.debug(f"Converged after {nr_iterations} iterations - {max_dx}")
                break

            if nr_iterations == self.max_iter:
                logging.debug(
                    f"Did not converge after {nr_iterations} iterations - {max_dx}"
                )
                break

        parameters_polynomial = parameters.copy()

        # Map parameters back to original basis
        parameters_polynomial = transform_parameters(
            self.Tampl,
            self.Tphase,
            parameters_polynomial,
            self.ampl_poly.nr_coeffs,
            self.phase_poly.nr_coeffs,
            self.nr_stations,
            self.nr_phase_updates,
        )

        # Reshape amplitude and slow phase parameters coefficient to match desired shape
        # amplitude parameters: reshaped into (nr_stations, 1, nr_parameters_ampl) array
        amplitude_shape = (self.nr_channel_blocks, self.nr_stations, 1, self.ampl_poly.nr_coeffs)
        amplitude1_coefficients = parameters_polynomial[
            :, :, : self.ampl_poly.nr_coeffs
        ].reshape(*amplitude_shape)
        amplitude2_coefficients = parameters_polynomial[
            :, :, self.ampl_poly.nr_coeffs : 2 * self.ampl_poly.nr_coeffs
        ].reshape(*amplitude_shape)
        slowphase1_coefficients = parameters_polynomial[
            :, :, 2 * self.ampl_poly.nr_coeffs : 3 * self.ampl_poly.nr_coeffs
        ].reshape(*amplitude_shape)
        slowphase2_coefficients = parameters_polynomial[
            :, :, 3 * self.ampl_poly.nr_coeffs : 4 * self.ampl_poly.nr_coeffs
        ].reshape(*amplitude_shape)
        # phase parameters: reshaped into (nr_stations, nr_phase_updates, nr_parameters_phase) array
        phase_coefficients = parameters_polynomial[
            :, :, 4 * self.ampl_poly.nr_coeffs : :
        ].reshape(
            self.nr_channel_blocks,
            self.nr_stations,
            self.nr_phase_updates,
            self.phase_poly.nr_coeffs,
        )

        offset_amplitude = (0, 0, self.count_process_buffer_calls, 0)
        offset_phase = (
            0,
            0,
            self.count_process_buffer_calls * self.nr_phase_updates,
            0,
        )
        self.h5writer.fill_solution_table(
            "amplitude1_coefficients", amplitude1_coefficients, offset_amplitude
        )
        self.h5writer.fill_solution_table(
            "amplitude2_coefficients", amplitude2_coefficients, offset_amplitude
        )
        self.h5writer.fill_solution_table(
            "slowphase1_coefficients", slowphase1_coefficients, offset_amplitude
        )
        self.h5writer.fill_solution_table(
            "slowphase2_coefficients", slowphase2_coefficients, offset_amplitude
        )
        self.h5writer.fill_solution_table(
            "phase_coefficients", phase_coefficients, offset_phase
        )

        if self.write_model_data:
            for channel_block, channel_block_frequencies in enumerate(self.frequencies):
                self.proxy.degridding(
                    self.kernel_size,
                    channel_block_frequencies,
                    visibilities,
                    uvw,
                    self.baselines,
                    aterms[channel_block],
                    self.aterm_offsets,
                    self.taper2,
                )
                for idx, dpbuffer in enumerate(self.dpbuffers):
                    visibilities_out = np.array(dpbuffer.get_data(), copy=False)
                    visibilities_out[
                        :,
                        channel_block
                        * self.nr_channels_per_block : (channel_block + 1)
                        * self.nr_channels_per_block,
                        :,
                    ] = visibilities[:, idx, :, :]

        self.count_process_buffer_calls += 1

    def __compute_amplitude(self, i, parameters):
        """
        Return the amplitude, given station index is and expansion coefficients

        Parameters
        ----------
        i : int
            Station index
        parameters : np.ndarray
            Array containing the expansion coefficients both for amplitude and phase

        Returns
        -------
        np.ndarray
            Array containing the complex exponential, shape is (nr_phase_updates, subgrid_size, subgrid_size, nr_correlations)
        """

        amplitude1 = np.tensordot(
            parameters[:, i, : self.ampl_poly.nr_coeffs],
            self.Bampl1,
            axes=((1,), (0,)),
        )
        amplitude2 = np.tensordot(
            parameters[:, i, self.ampl_poly.nr_coeffs : 2 * self.ampl_poly.nr_coeffs],
            self.Bampl2,
            axes=((1,), (0,)),
        )
        amplitude = amplitude1 + amplitude2

        # Result is repeated nr_phase_updates times to match complex exponential term
        return np.repeat(
            amplitude[:, np.newaxis, :],
            self.nr_phase_updates,
            axis=1,
        )

    def __compute_phase(self, i, parameters):
        """
        Return the complex exponential, given station index i and the expansion coefficients

        Parameters
        ----------
        i : int
            Station index
        parameters : np.ndarray
            Array containing the expansion coefficients both for amplitude and phase

        Returns
        -------
        np.ndarray
            Array containing the complex exponential, shape is (nr_phase_updates, subgrid_size, subgrid_size, nr_correlations)
        """

        slow_phase = np.tensordot(
            parameters[
                :, i, 2 * self.ampl_poly.nr_coeffs : 3 * self.ampl_poly.nr_coeffs
            ],
            self.Bslowphase1,
            axes=((1,), (0,)),
        ) + np.tensordot(
            parameters[
                :, i, 3 * self.ampl_poly.nr_coeffs : 4 * self.ampl_poly.nr_coeffs
            ],
            self.Bslowphase2,
            axes=((1,), (0,)),
        )

        return np.repeat(
            np.exp(1j * slow_phase[:, np.newaxis, :]),
            self.nr_phase_updates,
            axis=1,
        ) * np.exp(
            1j
            * np.tensordot(
                parameters[:, i, 4 * self.ampl_poly.nr_coeffs :].reshape(
                    (
                        self.nr_channel_blocks,
                        self.nr_phase_updates,
                        self.phase_poly.nr_coeffs,
                    )
                ),
                self.Bphase,
                axes=((2,), (0,)),
            )
        )


def compute_aterm_derivatives(
    aterm_ampl, aterm_phase, aterm_beam, B_a1, B_a2, B_sp1, B_sp2, B_p
):
    """
    Compute the partial derivatives of g = B_a*x_a * exp(j*B_p*x_p):
    - \partial g / \partial x_a = B_a * exp(j*B_p*x_p)
    - \partial g / \partial x_p = B_a * x_a * j * B_p * exp(j*B_p*x_p)
    where x_a and x_p the unknown expansion coefficients for the amplitude and
    the phase, respectively.

    Parameters
    ----------
    aterm_ampl : np.ndarray
        Amplitude tensor product B_a * x_a, should have shape (nr_phase_updates, subgrid_size, subgrid_size, nr_correlations)
    aterm_phase : np.ndarray
        Phase tensor product B_p * x_p, should have shape (nr_phase_updates, subgrid_size, subgrid_size, nr_correlations)
    aterm_beam : None or np.ndarray
        Beam to apply, should have shape (subgrid_size, subgrid_size, nr_correlations)
    B_a : np.ndarray
        Expanded (amplitude) basis functions, should have shape (nr_ampl_coeffs, subgrid_size, subgrid_size, nr_correlations)
    B_p : np.ndarray
        Expande (phase) basis functions, should have shape (nr_phase_coeffs, subgrid_size, subgrid_size, nr_correlations)

    Returns
    -------
    np.ndarray
        Column stacked derivative [\partial g/ \partial x_a, \partial g / \partial x_p]^T
        Output has shape (nr_phase_updates, nr_coeffs, subgrid_size, subgrid_size, nr_correlations)
        where nr_coeffs = len(x_a) + len(x_p)
    """
    # new-axis is introduced at "stations" axis
    aterm_derivatives_ampl1 = (
        aterm_phase[:, :, np.newaxis, :, :, :]
        * B_a1[np.newaxis, np.newaxis, :, :, :, :]
    )

    aterm_derivatives_ampl2 = (
        aterm_phase[:, :, np.newaxis, :, :, :]
        * B_a2[np.newaxis, np.newaxis, :, :, :, :]
    )

    aterm_derivatives_slowphase1 = (
        1j
        * aterm_ampl[:, :, np.newaxis, :, :, :]
        * aterm_phase[:, :, np.newaxis, :, :, :]
        * B_sp1[np.newaxis, np.newaxis, :, :, :, :]
    )

    aterm_derivatives_slowphase2 = (
        1j
        * aterm_ampl[:, :, np.newaxis, :, :, :]
        * aterm_phase[:, :, np.newaxis, :, :, :]
        * B_sp2[np.newaxis, np.newaxis, :, :, :, :]
    )

    aterm_derivatives_phase = (
        1j
        * aterm_ampl[:, :, np.newaxis, :, :, :]
        * aterm_phase[:, :, np.newaxis, :, :, :]
        * B_p[np.newaxis, np.newaxis, :, :, :, :]
    )

    aterm_derivatives = np.concatenate(
        (
            aterm_derivatives_ampl1,
            aterm_derivatives_ampl2,
            aterm_derivatives_slowphase1,
            aterm_derivatives_slowphase2,
            aterm_derivatives_phase,
        ),
        axis=2,
    )

    aterm_derivatives = apply_beam(aterm_beam, aterm_derivatives)

    aterm_derivatives = np.ascontiguousarray(aterm_derivatives, dtype=np.complex64)
    return aterm_derivatives


def transform_parameters(
    tmat_amplitude,
    tmat_phase,
    parameters,
    nr_amplitude_params,
    nr_phase_params,
    nr_stations,
    nr_timeslots,
):
    """
    Transform parameters (between orthonormalized and regular basis)

    Parameters
    ----------
    tmat_amplitude : np.ndarray
        Transformation matrix for the amplitudes
    tmat_phase: np.ndarray
        Transformation matrix for the phases
    parameters : np.ndarray
        Matrix with parameters
    nr_amplitude_params : int
        Number of amplitude parameters
    nr_phase_params : int
        Number of phase parameters
    nr_stations : int
        Number of stations
    nr_timeslots : int
        Number of timeslots
    """

    # Map the amplitudes
    for j in range(4):
        parameters[:, :, j * nr_amplitude_params : (j + 1) * nr_amplitude_params] = (
            parameters[:, :, j * nr_amplitude_params : (j + 1) * nr_amplitude_params]
            @ tmat_amplitude.T
        )

    # Map the phases
    for j in range(nr_timeslots):
        slicer = slice(
            4 * nr_amplitude_params + j * nr_phase_params,
            4 * nr_amplitude_params + (j + 1) * nr_phase_params,
        )
        parameters[:, :, slicer] = parameters[:, :, slicer] @ tmat_phase.T
    return parameters
