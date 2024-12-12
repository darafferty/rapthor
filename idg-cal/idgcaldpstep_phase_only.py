# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import everybeam as eb
import numpy as np
import idg
from idg.h5parmwriter import H5ParmWriter
from idg.basisfunctions import LagrangePolynomial
from idg.idgcalutils import (
    apply_beam,
    expand_basis_functions,
    next_composite,
    idgwindow,
    get_aterm_offsets,
    init_h5parm_solution_table,
)
from idg.idgcaldpstepbase import IDGCalDPStepBase
import astropy.io.fits as fits
import scipy.linalg
import time
import logging


class IDGCalDPStepPhaseOnly(IDGCalDPStepBase):
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

        ## BEGIN: read parset
        self.proxytype = parset.get_string(prefix + "proxytype", "CPU")

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
        ## END: read parset

        self.shift = np.array((0.0, 0.0), dtype=np.float32)

        # Number of phase updates per amplitude interval
        self.nr_phase_updates = self.ampl_interval // self.phase_interval

        # Initialize phase polynomial
        self.phase_poly = LagrangePolynomial(order=phase_order)

        self.nr_parameters0 = self.phase_poly.nr_coeffs
        self.nr_parameters = self.phase_poly.nr_coeffs * self.nr_phase_updates

        self.nr_channels_per_block = parset.get_int(prefix + "nr_channels_per_block", 0)
        self.apply_phase_constraint = parset.get_bool(
            prefix + "apply_phase_constraint", False
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

        # Initialize solution tables for phase coefficients
        # TODO: maybe pass parset as string to HISTORY?
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
        offset = (d.shape[-1] - N0) // 2
        d = d[:, :, offset : offset + N0, offset : offset + N0]
        d[np.isnan(d)]=0.0
        self.grid[0, (N - N0) // 2 : (N + N0) // 2, (N - N0) // 2 : (N + N0) // 2] = d[
            0, 0, :, :
        ] / np.outer(taper_grid0, taper_grid0)
        self.grid[3, (N - N0) // 2 : (N + N0) // 2, (N - N0) // 2 : (N + N0) // 2] = d[
            0, 0, :, :
        ] / np.outer(taper_grid0, taper_grid0)

        self.proxy.set_grid(self.grid)
        assert not np.any(np.isnan(self.grid))
        self.proxy.transform(idg.ImageDomainToFourierDomain)
        assert not np.any(np.isnan(self.grid))

        self.proxy.init_cache(
            self.subgrid_size, self.cell_size, self.w_step, self.shift
        )

        self.aterm_offsets = get_aterm_offsets(
            self.nr_phase_updates, self.ampl_interval
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

        # Initialize coefficients for phase to zeros.
        parameters = np.zeros(
            (self.nr_channel_blocks, self.nr_stations, self.nr_parameters)
        )

        # Map parameters to orthonormal basis
        parameters = transform_parameters(
            np.linalg.inv(self.Tphase),
            parameters,
            self.phase_poly.nr_coeffs,
            self.nr_phase_updates,
        )

        # Compute phase aterms from parameters

        aterm_phase = np.exp(
            1j
            * np.tensordot(
                parameters[:, :, :].reshape(
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
        ) * np.array([1.0, 0.0, 0.0, 1.0])

        # Transpose swaps the station and phase_update axes
        aterms = aterm_phase.transpose((0, 2, 1, 3, 4, 5))

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

                aterm_phase = self.__compute_phase(i, parameters)

                if aterms_beam is not None:
                    aterm_derivatives = compute_aterm_derivatives(
                        aterm_phase, aterms_beam[i], self.Bphase
                    )
                else:
                    aterm_derivatives = compute_aterm_derivatives(
                        aterm_phase, None, self.Bphase
                    )

                timer0 -= time.time()
                self.proxy.calibrate_update(
                    i, aterms, aterm_derivatives, hessian, gradient, residual
                )

                timer0 += time.time()
                residual_sum += residual[0]

                gradient = np.concatenate(
                    (gradient.reshape((self.nr_channel_blocks, -1)),), axis=1
                )

                H11 = np.concatenate(
                    [
                        scipy.linalg.block_diag(
                            *[
                                hessian[sb, t, :, :]
                                for t in range(self.nr_phase_updates)
                            ]
                        )[np.newaxis, :, :]
                        for sb in range(self.nr_channel_blocks)
                    ]
                )

                hessian = np.concatenate(
                    [H11[sb][np.newaxis, :, :] for sb in range(self.nr_channel_blocks)]
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
                    parameters[:, i, :] = self.constraint_matrix @ parameters[:, i, :]

                # Recompute aterms with updated parameters
                aterm_phase = self.__compute_phase(i, parameters)

                aterms_i = aterm_phase

                if aterms_beam is not None:
                    aterms_i = apply_beam(aterms_beam[i], aterms_i)

                aterms[:, :, i] = aterms_i
                timer1 += time.time()

            dresidual = previous_residual - residual_sum
            fractional_dresidual = dresidual / residual_sum

            logging.debug(residual_sum, max_dx, fractional_dresidual)

            previous_residual = residual_sum

            # converged = (nr_iterations > 1) and (fractional_dresidual < 1e-5)
            converged = (nr_iterations > 1) and (max_dx < 1e-2)

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
            self.Tphase,
            parameters_polynomial,
            self.phase_poly.nr_coeffs,
            self.nr_phase_updates,
        )

        # Reshape parameters, splitting last axis into two
        phase_coefficients = parameters_polynomial.reshape(
            self.nr_channel_blocks,
            self.nr_stations,
            self.nr_phase_updates,
            self.phase_poly.nr_coeffs,
        )

        offset_phase = (
            0,
            0,
            self.count_process_buffer_calls * self.nr_phase_updates,
            0,
        )
        self.h5writer.fill_solution_table(
            "phase_coefficients", phase_coefficients, offset_phase
        )
        self.count_process_buffer_calls += 1
        self.proxy.calibrate_finish()

    def __compute_phase(self, i, parameters):
        """
        Return the complex exponential, given station index i and the expansion coefficients

        Parameters
        ----------
        i : int
            Station index
        parameters : np.ndarray
            Array containing the expansion coefficients both for phase

        Returns
        -------
        np.ndarray
            Array containing the complex exponential, shape is (nr_phase_updates, subgrid_size, subgrid_size, nr_correlations)
        """

        return np.exp(
            1j
            * np.tensordot(
                parameters[:, i, :].reshape(
                    (
                        self.nr_channel_blocks,
                        self.nr_phase_updates,
                        self.phase_poly.nr_coeffs,
                    )
                ),
                self.Bphase,
                axes=((2,), (0,)),
            )
        ) * np.array([1.0, 0.0, 0.0, 1.0])


def compute_aterm_derivatives(aterm_phase, aterm_beam, B_p):
    """
    Compute the partial derivatives of g = B_a*x_a * exp(j*B_p*x_p):
    - \partial g / \partial x_p = B_a * x_a * j * B_p * exp(j*B_p*x_p)
    where x_p are the expansion coefficients for the phase.

    Parameters
    ----------
    aterm_phase : np.ndarray
        Phase tensor product B_p * x_p, should have shape (nr_phase_updates, subgrid_size, subgrid_size, nr_correlations)
    aterm_beam : None or np.ndarray
        Beam to apply, should have shape (subgrid_size, subgrid_size, nr_correlations)
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

    aterm_derivatives = (
        1j
        * aterm_phase[:, :, np.newaxis, :, :, :]
        * B_p[np.newaxis, np.newaxis, :, :, :, :]
    )

    aterm_derivatives = apply_beam(aterm_beam, aterm_derivatives)

    aterm_derivatives = np.ascontiguousarray(aterm_derivatives, dtype=np.complex64)
    return aterm_derivatives


def transform_parameters(
    tmat_phase,
    parameters,
    nr_phase_params,
    nr_timeslots,
):
    """
    Transform parameters (between orthonormalized and regular basis)

    Parameters
    ----------
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

    # Map the phases
    for j in range(nr_timeslots):
        slicer = slice(
            j * nr_phase_params,
            (j + 1) * nr_phase_params,
        )
        parameters[:, :, slicer] = parameters[:, :, slicer] @ tmat_phase.T
    return parameters
