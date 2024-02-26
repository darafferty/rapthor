# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import dp3
import everybeam as eb
import numpy as np
import idg
from idg.h5parmwriter import H5ParmWriter
from idg.basisfunctions import LagrangePolynomial
from idg.idgcalutils import apply_beam
from idg.idgcaldpstepbase import IDGCalDPStepBase
import astropy.io.fits as fits
import scipy.linalg
import time
import logging


class IDGCalDPStepScalar(IDGCalDPStepBase):
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
                self.nr_parameters - self.ampl_poly.nr_coeffs,
            )
        )

        parameters = np.concatenate((X0, X1, X2), axis=-1)
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
        aterm_ampl = np.tensordot(
            parameters[:, :, : self.ampl_poly.nr_coeffs], self.Bampl, axes=((2,), (0,))
        )
        aterm_phase = np.exp(
            1j
            * np.tensordot(
                parameters[:, :, self.ampl_poly.nr_coeffs :].reshape(
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
            * aterm_ampl[:, np.newaxis, :, :, :, :]
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
                        aterm_ampl, aterm_phase, aterms_beam[i], self.Bampl, self.Bphase
                    )
                else:
                    aterm_derivatives = compute_aterm_derivatives(
                        aterm_ampl, aterm_phase, None, self.Bampl, self.Bphase
                    )

                timer0 -= time.time()
                self.proxy.calibrate_update(
                    i, aterms, aterm_derivatives, hessian, gradient, residual
                )

                timer0 += time.time()
                residual0 = residual[0]
                residual_sum += residual[0]

                gradient = np.concatenate(
                    (
                        np.sum(gradient[:, :, : self.ampl_poly.nr_coeffs], axis=1),
                        gradient[:, :, self.ampl_poly.nr_coeffs :].reshape(
                            (self.nr_channel_blocks, -1)
                        ),
                    ),
                    axis=1,
                )

                H00 = hessian[
                    :, :, : self.ampl_poly.nr_coeffs, : self.ampl_poly.nr_coeffs
                ].sum(axis=1)
                H01 = np.concatenate(
                    [
                        hessian[
                            :, t, : self.ampl_poly.nr_coeffs, self.ampl_poly.nr_coeffs :
                        ]
                        for t in range(self.nr_phase_updates)
                    ],
                    axis=2,
                )
                H10 = np.concatenate(
                    [
                        hessian[
                            :, t, self.ampl_poly.nr_coeffs :, : self.ampl_poly.nr_coeffs
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
                                    self.ampl_poly.nr_coeffs :,
                                    self.ampl_poly.nr_coeffs :,
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
                hessian0 = hessian

                # Per channel_group, apply the inverse of the Hessian to the gradient
                # s is the channel_group index, ij are the rows and columns of the Hessian
                dx = np.einsum(
                    "sij,sj->si", np.linalg.pinv(hessian, self.pinv_tol), gradient
                )

                if max_dx < np.amax(abs(dx)):
                    max_dx = np.amax(abs(dx))
                    i_max = i

                parameters[:, i] += self.solver_update_gain * dx

                # If requested, apply constraint to phase parameters
                if self.apply_phase_constraint:
                    parameters[:, i, self.ampl_poly.nr_coeffs :] = (
                        self.constraint_matrix
                        @ parameters[:, i, self.ampl_poly.nr_coeffs :]
                    )

                # Recompute aterms with updated parameters
                aterm_ampl = self.__compute_amplitude(i, parameters)
                aterm_phase = self.__compute_phase(i, parameters)

                aterms_i = aterm_ampl * aterm_phase

                if aterms_beam is not None:
                    aterms_i = apply_beam(aterms_beam[i], aterms_i)

                aterms[:, :, i] = aterms_i
                timer1 += time.time()

            dresidual = previous_residual - residual_sum
            fractional_dresidual = dresidual / residual_sum

            logging.debug(max_dx, fractional_dresidual)

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

        # Reshape amplitude/parameters coefficient to match desired shape
        # amplitude parameters: reshaped into (nr_stations, 1, nr_parameters_ampl) array
        amplitude_coefficients = parameters_polynomial[
            :, :, : self.ampl_poly.nr_coeffs
        ].reshape(self.nr_channel_blocks, self.nr_stations, 1, self.ampl_poly.nr_coeffs)
        # phase parameters: reshaped into (nr_stations, nr_phase_updates, nr_parameters_phase) array
        phase_coefficients = parameters_polynomial[
            :, :, self.ampl_poly.nr_coeffs : :
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
            "amplitude_coefficients", amplitude_coefficients, offset_amplitude
        )
        self.h5writer.fill_solution_table(
            "phase_coefficients", phase_coefficients, offset_phase
        )
        self.count_process_buffer_calls += 1

    def __compute_amplitude(self, i, parameters):
        """
        Return the amplitude, given station index i and expansion coefficients

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
        # Result is repeated nr_phase_updates times to match complex exponential term
        return np.repeat(
            np.tensordot(
                parameters[:, i, : self.ampl_poly.nr_coeffs],
                self.Bampl,
                axes=((1,), (0,)),
            )[:, np.newaxis, :],
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
        return np.exp(
            1j
            * np.tensordot(
                parameters[:, i, self.ampl_poly.nr_coeffs :].reshape(
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


def compute_aterm_derivatives(aterm_ampl, aterm_phase, aterm_beam, B_a, B_p):
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
    aterm_derivatives_ampl = (
        aterm_phase[:, :, np.newaxis, :, :, :] * B_a[np.newaxis, np.newaxis, :, :, :, :]
    )

    aterm_derivatives_phase = (
        1j
        * aterm_ampl[:, :, np.newaxis, :, :, :]
        * aterm_phase[:, :, np.newaxis, :, :, :]
        * B_p[np.newaxis, np.newaxis, :, :, :, :]
    )

    aterm_derivatives = np.concatenate(
        (aterm_derivatives_ampl, aterm_derivatives_phase), axis=2
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
    parameters[:, :, :nr_amplitude_params] = (
        parameters[:, :, :nr_amplitude_params] @ tmat_amplitude.T
    )

    # Map the phases
    for j in range(nr_timeslots):
        slicer = slice(
            nr_amplitude_params + j * nr_phase_params,
            nr_amplitude_params + (j + 1) * nr_phase_params,
        )
        parameters[:, :, slicer] = parameters[:, :, slicer] @ tmat_phase.T
    return parameters
