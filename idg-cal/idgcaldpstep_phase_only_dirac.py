# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import everybeam as eb
import numpy as np
import idg
from idg.h5parmwriter import H5ParmWriter
from idg.basisfunctions import LagrangePolynomial
from idg.idgcalutils import (
    apply_beam,
    next_composite,
    idgwindow,
    get_aterm_offsets,
    init_h5parm_solution_table,
)
from idg.idgcaldpstep_phase_only import IDGCalDPStepPhaseOnly
import astropy.io.fits as fits
from idg.lbfgsb import LBFGSB
import time
import logging
from matplotlib import pyplot as plt


class IDGCalDPStepPhaseOnlyDirac(IDGCalDPStepPhaseOnly):
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
        # parameters = np.random.default_rng().uniform(-np.pi,np.pi,(self.nr_channel_blocks, self.nr_stations, self.nr_parameters))

        # Map parameters to orthonormal basis
        parameters = transform_parameters(
            np.linalg.inv(self.Tphase),
            parameters,
            self.phase_poly.nr_coeffs,
            self.nr_phase_updates,
        )

#################################################################################
        def lbfgs_cost_function(parameters):

            parameters = parameters.reshape(
                (self.nr_channel_blocks, self.nr_stations, self.nr_parameters)
            )
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
            aterms = np.swapaxes(aterm_phase, 1, 2)

            aterms = apply_beam(aterms_beam, aterms)

            aterms = np.ascontiguousarray(aterms.astype(idg.idgtypes.atermtype))

            residual = np.zeros((self.nr_channel_blocks,), dtype=np.float64)

            r = 0
            for i in range(self.nr_stations):
                aterm_phase = self.__compute_phase(i, parameters)
                if aterms_beam is not None:
                    aterm_derivatives = compute_aterm_derivatives(
                        aterm_phase, aterms_beam[i], self.Bphase
                    )
                else:
                    aterm_derivatives = compute_aterm_derivatives(
                        aterm_phase, None, self.Bphase
                    )

                residual[:] = 0.0
                self.proxy.calc_cost(
                    i, aterms, aterm_derivatives, residual 
                )
                r += residual[0] / 2

            residual = r

            return residual

        def lbfgs_grad_function(parameters):

            parameters = parameters.reshape(
                (self.nr_channel_blocks, self.nr_stations, self.nr_parameters)
            )
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
            aterms = np.swapaxes(aterm_phase, 1, 2)

            aterms = apply_beam(aterms_beam, aterms)

            aterms = np.ascontiguousarray(aterms.astype(idg.idgtypes.atermtype))

            gradient = np.zeros(
                (self.nr_channel_blocks, self.nr_phase_updates, self.nr_parameters0),
                dtype=np.float64,
            )

            g = []
            for i in range(self.nr_stations):
                aterm_phase = self.__compute_phase(i, parameters)
                if aterms_beam is not None:
                    aterm_derivatives = compute_aterm_derivatives(
                        aterm_phase, aterms_beam[i], self.Bphase
                    )
                else:
                    aterm_derivatives = compute_aterm_derivatives(
                        aterm_phase, None, self.Bphase
                    )

                gradient[:] = 0.0
                self.proxy.calc_gradient(
                    i, aterms, aterm_derivatives, gradient
                )
                g.append(-2 * gradient.flatten())

            # also add column dimension
            gradient = np.expand_dims(np.concatenate(g),-1)

            return gradient

#################################################################################

        x0 = np.expand_dims(parameters.flatten(),-1)
        # lower and upper bounds
        x_low = np.ones(x0.shape)*(-1000.0)
        x_high = np.ones(x0.shape)*(1000.0)

        optimizer = LBFGSB(x0, x_low, x_high, max_iter=self.max_iter)
        result = optimizer.step(lbfgs_cost_function, lbfgs_grad_function)

        logging.debug(result['residual'], result['success'])

        parameters = result['x'].reshape(
            (self.nr_channel_blocks, self.nr_stations, self.nr_parameters)
        )
        parameters_polynomial = parameters.copy()

        # Map parameters back to original basis
        parameters_polynomial = transform_parameters(
            self.Tphase,
            parameters_polynomial,
            self.phase_poly.nr_coeffs,
            self.nr_phase_updates,
        )

        # Reshape phase coefficients to match desired shape
        # phase parameters: reshaped into (nr_stations, nr_phase_updates, nr_parameters_phase) array
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

    def plot_residual(self, i, visibilities, weights, uvw, aterms, parameters):
        logging.debug("plot_residual ", i)
        visibilities_predicted = visibilities.copy()

        aterms1 = aterms.copy()
        parameters1 = parameters.copy()

        residuals = []
        x = []
        for dx in np.linspace(-1.0, 1.0, 101):
            parameters1[:, i] = parameters[:, i] + dx
            x.append(parameters[:, i, 0] + dx)
            aterms_i = self.__compute_phase(i, parameters1)
            aterms1[:, :, i] = aterms_i
            self.proxy.degridding(
                self.kernel_size,
                self.frequencies[0],
                visibilities_predicted,
                uvw,
                self.baselines,
                aterms1[0],
                self.aterm_offsets,
                self.taper2,
            )
            residuals.append(
                np.sum(np.abs(visibilities - visibilities_predicted) ** 2 * weights)
            )
        plt.clf()
        plt.plot(x, residuals)
        plt.savefig(f"residuals{i}.png")

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
    Compute the partial derivatives of g = exp(j*B_p*x_p):
    - \partial g / \partial x_p = B_a * x_a * j * B_p * exp(j*B_p*x_p)
    where x_a and x_p the unknown expansion coefficients for the phase.

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
