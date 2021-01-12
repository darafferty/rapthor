# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import dppp
import numpy as np
import idg
import idg.util
import astropy.io.fits as fits
import scipy.linalg
import time


class IDGCalDPStep(dppp.DPStep):
    def __init__(self, parset, prefix):
        super().__init__()
        self.read_parset(parset, prefix)
        self.dpbuffers = []
        self.is_initialized = False

    def show(self):
        print()
        print("IDGCalDPStep")

    def process(self, dpbuffer):

        # Accumulate buffers
        self.dpbuffers.append(dpbuffer)

        # If we have accumulated enough data, process it
        if len(self.dpbuffers) == self.nr_timesteps:
            self.process_buffers()

            # Send processed data to the next step
            for dpbuffer in self.dpbuffers:
                self.process_next_step(dpbuffer)

            # Clear accumulated data
            self.dpbuffers = []

    def finish(self):
        # If there is any remaining data, process it
        if len(self.dpbuffers):
            # TODO deal with incomplete solution interval
            # self.process_buffers()
            for dpbuffer in self.dpbuffers:
                self.process_next_step(dpbuffer)
            self.dpbuffers = []

    def update_info(self, dpinfo):
        super().update_info(dpinfo)
        self.info().set_need_vis_data()
        self.fetch_uvw = True
        self.fetch_weights = True

    def read_parset(self, parset, prefix):
        """
        Read relevant information from a given parset

        Parameters
        ----------
        parset : dppp.ParameterSet
            ParameterSet object provided by DP3
        prefix : str
            Prefix to be used when reading the parset.
        """
        # TODO: self.parset / self.prefix
        # should be local to this method
        self.parset = parset
        self.prefix = prefix

        solint = parset.getInt(prefix + "solint", 0)
        if solint:
            self.solution_interval_amplitude = solint
            self.solution_interval_phase = solint
        else:
            self.solution_interval_amplitude = parset.getInt(
                prefix + "solintamplitude", 0
            )
            self.solution_interval_phase = parset.getInt(prefix + "solintphase", 0)

        self.imagename = parset.getString(prefix + "modelimage")
        self.padding = parset.getFloat(prefix + "padding", 1.2)

        self.nr_timeslots = 40
        self.nr_timesteps_per_slot = 4

        self.nr_timesteps = self.nr_timeslots * self.nr_timesteps_per_slot

        self.nr_correlations = 4
        self.subgrid_size = 32

        self.taper_support = 7
        self.wterm_support = 5
        self.aterm_support = 5

        self.kernel_size = self.taper_support + self.wterm_support + self.aterm_support

        self.nr_parameters_ampl = 6
        self.nr_parameters_phase = 3
        self.nr_parameters0 = self.nr_parameters_ampl + self.nr_parameters_phase
        self.nr_parameters = (
            self.nr_parameters_ampl + self.nr_parameters_phase * self.nr_timeslots
        )

        self.solver_update_gain = 0.5
        self.pinv_tol = 1e-9
        self.max_iter = 1

        self.w_step = 400.0

    def initialize(self):
        self.is_initialized = True

        self.nr_stations = self.info().nantenna()
        self.nr_baselines = (self.nr_stations * (self.nr_stations - 1)) // 2
        self.frequencies = np.array(
            self.info().get_channel_frequencies(), dtype=np.float32
        )
        self.nr_channels = len(self.frequencies)
        self.baselines = np.zeros(shape=(self.nr_baselines), dtype=idg.baselinetype)

        station1 = np.array(self.info().get_antenna1())
        station2 = np.array(self.info().get_antenna2())
        self.auto_corr_mask = station1 != station2
        self.baselines["station1"] = station1[self.auto_corr_mask]
        self.baselines["station2"] = station2[self.auto_corr_mask]

        # initialize proxy
        # self.proxy = idg.HybridCUDA.GenericOptimized(self.nr_correlations, self.subgrid_size)
        self.proxy = idg.CPU.Optimized(self.nr_correlations, self.subgrid_size)

        # read image dimensions from fits header
        h = fits.getheader(self.imagename)
        N0 = h["NAXIS1"]
        self.cell_size = abs(h["CDELT1"]) / 180 * np.pi

        # compute padded image size
        N = next_composite(int(N0 * self.padding))
        self.grid_size = N
        self.image_size = N * self.cell_size

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
        ] = (
            taper_
            * np.exp(
                -1j
                * np.linspace(-np.pi / 2, np.pi / 2, self.subgrid_size, endpoint=False)
            )
        )
        taper_grid = (
            np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(taper_grid))).real
            * self.grid_size
            / self.subgrid_size
        )
        taper_grid0 = taper_grid[(N - N0) // 2 : (N + N0) // 2]

        # read image data, assume Stokes I
        d = fits.getdata(self.imagename)
        self.grid[0, (N - N0) // 2 : (N + N0) // 2, (N - N0) // 2 : (N + N0) // 2] = d[
            0, 0, :, :
        ] / np.outer(taper_grid0, taper_grid0)
        self.grid[3, (N - N0) // 2 : (N + N0) // 2, (N - N0) // 2 : (N + N0) // 2] = d[
            0, 0, :, :
        ] / np.outer(taper_grid0, taper_grid0)

        self.proxy.transform(idg.ImageDomainToFourierDomain, self.grid)

        self.shift = np.array((0.0, 0.0, 0.0), dtype=np.float32)

        self.aterms_offsets = idg.util.get_example_aterms_offset(
            self.nr_timeslots, self.nr_timesteps
        )

        self.Bampl, self.Tampl = polynomial_basis_functions(
            self.subgrid_size, self.image_size, self.nr_parameters_ampl
        )
        self.Bphase, self.Tphase = polynomial_basis_functions(
            self.subgrid_size, self.image_size, self.nr_parameters_phase
        )

    def process_buffers(self):
        """
        Processing the buffers. This is the central method within any class that
        derives from dppp.DPStep
        """

        if not self.is_initialized:
            self.initialize()

        # Concatenate accumulated data and display just the shapes
        visibilities = np.stack(
            [np.array(dpbuffer.get_data(), copy=False) for dpbuffer in self.dpbuffers],
            axis=1,
        )
        visibilities = visibilities[self.auto_corr_mask, :, :]
        flags = np.stack(
            [np.array(dpbuffer.get_flags(), copy=False) for dpbuffer in self.dpbuffers],
            axis=1,
        )
        flags = flags[self.auto_corr_mask, :, :]
        weights = np.stack(
            [
                np.array(dpbuffer.get_weights(), copy=False)
                for dpbuffer in self.dpbuffers
            ],
            axis=1,
        )
        weights = weights[self.auto_corr_mask, :, :]
        uvw_ = np.stack(
            [np.array(dpbuffer.get_uvw(), copy=False) for dpbuffer in self.dpbuffers],
            axis=1,
        )
        uvw = np.zeros(shape=(self.nr_baselines, self.nr_timesteps), dtype=idg.uvwtype)

        print(f"    shape uvw_: {uvw_.shape}")
        print(self.auto_corr_mask.shape)

        print(uvw_[self.auto_corr_mask, :, 0].shape)
        uvw["u"] = uvw_[self.auto_corr_mask, :, 0]
        uvw["v"] = -uvw_[self.auto_corr_mask, :, 1]
        uvw["w"] = -uvw_[self.auto_corr_mask, :, 2]

        print(f"    shape of baselines: {self.baselines.shape}")
        print(f"    shape of visibilities: {visibilities.shape}")
        print(f"    shape of flags: {flags.shape}")
        print(f"    shape of weights: {weights.shape}")
        print(f"    shape uvw: {uvw.shape}")

        weights *= ~flags

        self.proxy.calibrate_init(
            self.w_step,
            self.shift,
            self.cell_size,
            self.kernel_size,
            self.subgrid_size,
            self.frequencies,
            visibilities,
            weights,
            uvw,
            self.baselines,
            self.grid,
            self.aterms_offsets,
            self.taper2,
        )

        X0 = np.ones((self.nr_stations, 1))
        X1 = np.zeros((self.nr_stations, 1))

        # initialize parameters
        parameters = np.concatenate((X0,) + (self.nr_parameters - 1) * (X1,), axis=1)

        for i in range(self.nr_stations):
            parameters[i, : self.nr_parameters_ampl] = np.dot(
                np.linalg.inv(self.Tampl), parameters[i, : self.nr_parameters_ampl]
            )
            for j in range(self.nr_timeslots):
                parameters[
                    i,
                    self.nr_parameters_ampl
                    + j * self.nr_parameters_phase : self.nr_parameters_ampl
                    + (j + 1) * self.nr_parameters_phase,
                ] = np.dot(
                    np.linalg.inv(self.Tphase),
                    parameters[
                        i,
                        self.nr_parameters_ampl
                        + j * self.nr_parameters_phase : self.nr_parameters_ampl
                        + (j + 1) * self.nr_parameters_phase,
                    ],
                )

        aterms = idg.util.get_identity_aterms(
            self.nr_timeslots, self.nr_stations, self.subgrid_size, self.nr_correlations
        )
        aterms_offsets = idg.util.get_example_aterms_offset(
            self.nr_timeslots, self.nr_timesteps
        )

        aterm_ampl = np.tensordot(
            parameters[:, : self.nr_parameters_ampl], self.Bampl, axes=((1,), (0,))
        )
        aterm_phase = np.exp(
            1j
            * np.tensordot(
                parameters[:, self.nr_parameters_ampl :].reshape(
                    (self.nr_stations, self.nr_timeslots, self.nr_parameters_phase)
                ),
                self.Bphase,
                axes=((2,), (0,)),
            )
        )
        aterms[:, :, :, :, :] = aterm_phase.transpose((1, 0, 2, 3, 4)) * aterm_ampl

        nr_iterations = 0
        converged = False

        max_dx = 0.0

        timer = -time.time()

        timer0 = 0
        timer1 = 0

        previous_residual = 0.0

        while True:

            nr_iterations += 1

            print(f"iteration nr {nr_iterations} ")

            max_dx = 0.0
            norm_dx = 0.0
            residual_sum = 0.0
            for i in range(self.nr_stations):
                print(f"   {i}")
                timer1 -= time.time()

                # Predict visibilities for current solution
                hessian = np.zeros(
                    (self.nr_timeslots, self.nr_parameters0, self.nr_parameters0),
                    dtype=np.float64,
                )
                gradient = np.zeros(
                    (self.nr_timeslots, self.nr_parameters0), dtype=np.float64
                )
                residual = np.zeros((1,), dtype=np.float64)

                aterm_ampl = np.repeat(
                    np.tensordot(
                        parameters[i, : self.nr_parameters_ampl],
                        self.Bampl,
                        axes=((0,), (0,)),
                    )[np.newaxis, :],
                    self.nr_timeslots,
                    axis=0,
                )
                aterm_phase = np.exp(
                    1j
                    * np.tensordot(
                        parameters[i, self.nr_parameters_ampl :].reshape(
                            (self.nr_timeslots, self.nr_parameters_phase)
                        ),
                        self.Bphase,
                        axes=((1,), (0,)),
                    )
                )

                aterm_derivatives_ampl = (
                    aterm_phase[:, np.newaxis, :, :, :]
                    * self.Bampl[np.newaxis, :, :, :, :]
                )

                aterm_derivatives_phase = (
                    1j
                    * aterm_ampl[:, np.newaxis, :, :, :]
                    * aterm_phase[:, np.newaxis, :, :, :]
                    * self.Bphase[np.newaxis, :, :, :, :]
                )

                aterm_derivatives = np.concatenate(
                    (aterm_derivatives_ampl, aterm_derivatives_phase), axis=1
                )
                aterm_derivatives = np.ascontiguousarray(
                    aterm_derivatives, dtype=np.complex64
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
                        np.sum(gradient[:, : self.nr_parameters_ampl], axis=0),
                        gradient[:, self.nr_parameters_ampl :].flatten(),
                    )
                )

                for t in range(self.nr_timeslots):
                    print(hessian[t, :, :])
                H00 = hessian[
                    :, : self.nr_parameters_ampl, : self.nr_parameters_ampl
                ].sum(axis=0)
                H01 = np.concatenate(
                    [
                        hessian[t, : self.nr_parameters_ampl, self.nr_parameters_ampl :]
                        for t in range(self.nr_timeslots)
                    ],
                    axis=1,
                )
                H10 = np.concatenate(
                    [
                        hessian[t, self.nr_parameters_ampl :, : self.nr_parameters_ampl]
                        for t in range(self.nr_timeslots)
                    ],
                    axis=0,
                )
                H11 = scipy.linalg.block_diag(
                    *[
                        hessian[t, self.nr_parameters_ampl :, self.nr_parameters_ampl :]
                        for t in range(self.nr_timeslots)
                    ]
                )

                hessian = np.block([[H00, H01], [H10, H11]])
                hessian0 = hessian

                dx = np.dot(np.linalg.pinv(hessian, self.pinv_tol), gradient)
                # TODO: norm_dx (formally a squared-norm) not used?
                norm_dx += np.linalg.norm(dx) ** 2

                if max_dx < np.amax(abs(dx)):
                    max_dx = np.amax(abs(dx))
                    i_max = i

                p0 = parameters[i].copy()

                parameters[i] += self.solver_update_gain * dx

                aterm_ampl = np.repeat(
                    np.tensordot(
                        parameters[i, : self.nr_parameters_ampl],
                        self.Bampl,
                        axes=((0,), (0,)),
                    )[np.newaxis, :],
                    self.nr_timeslots,
                    axis=0,
                )
                aterm_phase = np.exp(
                    1j
                    * np.tensordot(
                        parameters[i, self.nr_parameters_ampl :].reshape(
                            (self.nr_timeslots, self.nr_parameters_phase)
                        ),
                        self.Bphase,
                        axes=((1,), (0,)),
                    )
                )

                aterms0 = aterms.copy()
                aterms[:, i] = aterm_ampl * aterm_phase

                timer1 += time.time()

            dresidual = previous_residual - residual_sum
            fractional_dresidual = dresidual / residual_sum

            print(max_dx, fractional_dresidual)

            previous_residual = residual_sum

            converged = (nr_iterations > 1) and (fractional_dresidual < 1e-6)

            if converged:
                print(f"Converged after {nr_iterations} iterations - {max_dx}")
                break

            if nr_iterations == self.max_iter:
                print(f"Did not converge after {nr_iterations} iterations - {max_dx}")
                break

        parameters_polynomial = parameters.copy()

        for i in range(self.nr_stations):
            parameters_polynomial[i, : self.nr_parameters_ampl] = np.dot(
                self.Tampl, parameters_polynomial[i, : self.nr_parameters_ampl]
            )
            for j in range(self.nr_timeslots):
                parameters_polynomial[
                    i,
                    self.nr_parameters_ampl
                    + j * self.nr_parameters_phase : self.nr_parameters_ampl
                    + (j + 1) * self.nr_parameters_phase,
                ] = np.dot(
                    self.Tphase,
                    parameters_polynomial[
                        i,
                        self.nr_parameters_ampl
                        + j * self.nr_parameters_phase : self.nr_parameters_ampl
                        + (j + 1) * self.nr_parameters_phase,
                    ],
                )


def polynomial_basis_functions(polynomial_order, subgrid_size, image_size):
    """
    Compute the (orthonormalized) Lagrange polynomial basis function on a
    given subgrid.

    Parameters
    ----------
    polynomial_order : int
        Polynomial order to be used in the expansion.
    subgrid_size : int
        Size of IDG subgrid (assumed to be square)
    image_size : int
        Size of image (assumed to be square)

    Returns
    -------
    np.ndarray, np.ndarray
        np.ndarray with evaluation of orthonormal basis functions and TODO (WHAT IS T)?
    """
    s = image_size / subgrid_size * (subgrid_size - 1)
    l = s * np.linspace(-0.5, 0.5, subgrid_size)
    m = -s * np.linspace(-0.5, 0.5, subgrid_size)

    B1, B2 = np.meshgrid(l, m)

    B1 = B1[np.newaxis, :, :, np.newaxis]
    B2 = B2[np.newaxis, :, :, np.newaxis]

    nr_terms = np.sum(np.arange(1, polynomial_order + 2, 1))
    basis_functions = np.empty((nr_terms,) + B1.shape[1::])

    for n in range(polynomial_order + 1):
        # Loop over polynomial degree (rows in Pascal's triangle)
        for k in range(n + 1):
            # Loop over unique entries per polynomial degree
            offset = np.sum(np.arange(1, n + 1, 1)) + k
            basis_functions[offset, ...] = B1 ** (n - k) * B2 ** k

    basis_functions = basis_functions.reshape((-1, subgrid_size * subgrid_size)).T
    U, S, V, = np.linalg.svd(basis_functions)
    basis_functions_orthonormal = U[:, :nr_terms]
    T = np.dot(np.linalg.pinv(basis_functions), basis_functions_orthonormal)
    basis_functions_orthonormal = basis_functions_orthonormal.T.reshape(
        (-1, subgrid_size, subgrid_size, 1)
    )

    basis_functions_orthonormal = np.kron(
        basis_functions_orthonormal, np.array([1.0, 0.0, 0.0, 1.0])
    )
    return basis_functions_orthonormal, T


def next_composite(n):
    n += n & 1
    while True:
        nn = n
        while (nn % 2) == 0:
            nn /= 2
        while (nn % 3) == 0:
            nn /= 3
        while (nn % 5) == 0:
            nn /= 5
        if nn == 1:
            return n
        n += 2


def idgwindow(N, W, padding, offset=0.5, l_range=None):
    """
    TODO: Documentation goes here

    Parameters
    ----------
    N : [type]
        [description]
    W : [type]
        [description]
    padding : [type]
        [description]
    offset : float, optional
        [description], by default 0.5
    l_range : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """

    l_range_inner = np.linspace(-(1 / padding) / 2, (1 / padding) / 2, N * 16 + 1)

    vl = (np.arange(N) - N / 2 + offset) / N
    vu = np.arange(N) - N / 2 + offset
    Q = np.sinc((N - W + 1) * (vl[:, np.newaxis] - vl[np.newaxis, :]))

    B = []
    RR = []
    for l in l_range_inner:
        d = np.mean(
            np.exp(2 * np.pi * 1j * vu[np.newaxis, :] * (vl[:, np.newaxis] - l)), axis=1
        ).real
        D = d[:, np.newaxis] * d[np.newaxis, :]
        b_avg = np.sinc((N - W + 1) * (l - vl))
        B.append(b_avg * d)
        S = b_avg[:, np.newaxis] * b_avg[np.newaxis, :]
        RR.append(D * (Q - S))
    B = np.array(B)
    RR = np.array(RR)

    taper = np.ones(len(l_range_inner))

    for q in range(10):
        R = np.sum((RR * 1 / taper[:, np.newaxis, np.newaxis] ** 2), axis=0)
        R1 = R[:, : (N // 2)] + R[:, : (N // 2) - 1 : -1]
        R2 = R1[: (N // 2), :] + R1[: (N // 2) - 1 : -1, :]
        U, S1, V = np.linalg.svd(R2)
        a = np.abs(np.concatenate([U[:, -1], U[::-1, -1]]))
        taper = np.dot(B, a)

    if l_range is None:
        return a
    else:
        B = []
        RR = []
        for l in l_range:
            d = np.mean(
                np.exp(2 * np.pi * 1j * vu[np.newaxis, :] * (vl[:, np.newaxis] - l)),
                axis=1,
            ).real
            D = d[:, np.newaxis] * d[np.newaxis, :]
            b_avg = np.sinc((N - W + 1) * (l - vl))
            B.append(b_avg * d)
            S = b_avg[:, np.newaxis] * b_avg[np.newaxis, :]
            RR.append(D * (Q - S))
        B = np.array(B)
        RR = np.array(RR)

        return a, B, RR


def main():
    subgrid_size = 32
    image_size = 1024
    # n_terms = 10
    polynomial_degree = 3
    polynomial_basis_functions(polynomial_degree, subgrid_size, image_size)


if __name__ == "__main__":
    main()
