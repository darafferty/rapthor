# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import dp3
import numpy as np
import idg
from idg.h5parmwriter import H5ParmWriter
from idg.basisfunctions import LagrangePolynomial
from idg.idgcalutils import next_composite, idgwindow, get_aterm_offsets, init_h5parm_solution_table
import astropy.io.fits as fits
import scipy.linalg
import time
import logging


class IDGCalDPStep(dp3.Step):
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
        if len(self.dpbuffers) == self.ampl_interval:
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
        parset : dp3.ParameterSet
            ParameterSet object provided by DP3
        prefix : str
            Prefix to be used when reading the parset.
        """

        ## BEGIN: read parset
        self.proxytype = parset.getString(prefix + "proxytype", "CPU")

        solint = parset.getInt(prefix + "solint", 0)
        if solint:
            self.ampl_interval = solint
            self.phase_interval = solint
        else:
            self.ampl_interval = parset.getInt(prefix + "solintamplitude", 0)
            self.phase_interval = parset.getInt(prefix + "solintphase", 0)

        # solintamplitude should be divisible by solintphase, check and correct if that's not the case
        remainder = self.ampl_interval % self.phase_interval
        if remainder != 0:
            logging.warning(
                f"Specified amplitude solution interval {self.ampl_interval} is not an integer multiple of the phase solution interval {self.phase_interval}. Amplitude soluton interval will be modified to {self.ampl_interval + remainder}"
            )
            self.ampl_interval += remainder

        self.imagename = parset.getString(prefix + "modelimage")
        self.padding = parset.getFloat(prefix + "padding", 1.2)
        self.nr_correlations = parset.getInt(prefix + "nrcorrelations", 4)
        self.subgrid_size = parset.getInt(prefix + "subgridsize", 32)

        self.taper_support = parset.getInt(prefix + "tapersupport", 7)
        wterm_support = parset.getInt(prefix + "wtermsupport", 5)
        aterm_support = parset.getInt(prefix + "atermsupport", 5)
        self.kernel_size = self.taper_support + wterm_support + aterm_support

        # get polynomial degrees for amplitude/phase
        ampl_order = parset.getInt(prefix + "polynomialdegamplitude", 2)
        phase_order = parset.getInt(prefix + "polynomialdegphase", 1)

        # Solver related
        # Factor between 0 and 1 with which to update solution between iterations
        self.solver_update_gain = parset.getFloat(prefix + "solverupdategain", 0.5)
        # Tolerance pseudo inverse
        self.pinv_tol = parset.getDouble(prefix + "tolerancepinv", 1e-9)
        # Maximum number of iterations
        self.max_iter = parset.getInt(prefix + "maxiter", 1)

        # H5Parm output file related
        self.h5parm_fname = parset.getString(prefix + "h5parm", "idgcal.h5")
        self.h5parm_solsetname = parset.getString(
            prefix + "h5parmsolset", "coefficients000"
        )
        self.h5parm_overwrite = parset.getBool(prefix + "h5parmoverwrite", True)

        self.w_step = parset.getFloat(prefix + "wstep", 400.0)
        ## END: read parset

        self.shift = np.array((0.0, 0.0, 0.0), dtype=np.float32)

        # Number of phase updates per amplitude interval
        self.nr_phase_updates = self.ampl_interval // self.phase_interval

        # Initialize amplitude and phase polynomial
        self.ampl_poly = LagrangePolynomial(order=ampl_order)
        self.phase_poly = LagrangePolynomial(order=phase_order)

        self.nr_parameters0 = self.ampl_poly.nr_coeffs + self.phase_poly.nr_coeffs
        self.nr_parameters = (
            self.ampl_poly.nr_coeffs + self.phase_poly.nr_coeffs * self.nr_phase_updates
        )

    def initialize(self):
        self.is_initialized = True

        # Counter for the number of calls to process_buffers
        self.count_process_buffer_calls = 0

        # Extract the time info and cast into a time array
        tstart = self.info().start_time()
        # Time array should match "amplitude time blocks"
        nsteps = (self.info().ntime() // self.ampl_interval) * self.ampl_interval
        dt = self.info().time_interval()
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

        self.nr_stations = self.info().nantenna()
        self.nr_baselines = (self.nr_stations * (self.nr_stations - 1)) // 2
        self.frequencies = np.array(
            self.info().get_channel_frequencies(), dtype=np.float32
        )
        self.nr_channels = len(self.frequencies)
        self.baselines = np.zeros(shape=(self.nr_baselines,2 ), dtype=np.int32)

        station1 = np.array(self.info().get_antenna1())
        station2 = np.array(self.info().get_antenna2())
        self.auto_corr_mask = station1 != station2
        self.baselines[:,0] = station1[self.auto_corr_mask]
        self.baselines[:,1] = station2[self.auto_corr_mask]

        # Axes data
        axes_labels = ["ant", "time", "dir"]
        axes_data_amplitude = dict(
            zip(
                axes_labels,
                (self.nr_stations, self.time_array_ampl.size, self.ampl_poly.nr_coeffs),
            )
        )
        axes_data_phase = dict(
            zip(
                axes_labels,
                (
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
        self.h5writer.add_antennas(
            self.info().antenna_names(), self.info().antenna_positions()
        )

        if self.proxytype.lower() == "gpu":
            self.proxy = idg.HybridCUDA.GenericOptimized(
                self.nr_correlations, self.subgrid_size
            )
        else:
            self.proxy = idg.CPU.Optimized()

        # read image dimensions from fits header
        h = fits.getheader(self.imagename)
        N0 = h["NAXIS1"]
        self.cell_size = abs(h["CDELT1"]) / 180 * np.pi

        # compute padded image size
        N = next_composite(int(N0 * self.padding))
        self.grid_size = N
        self.image_size = N * self.cell_size

        # Initialize solution tables for amplitude and phase coefficients
        # TODO: maybe pass parset as string to HISTORY?
        init_h5parm_solution_table(
            self.h5writer,
            "amplitude",
            axes_data_amplitude,
            self.info().antenna_names(),
            self.time_array_ampl,
            self.image_size,
            self.subgrid_size,
        )
        init_h5parm_solution_table(
            self.h5writer,
            "phase",
            axes_data_phase,
            self.info().antenna_names(),
            self.time_array_phase,
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

        self.proxy.set_grid(self.grid)
        self.proxy.transform(idg.ImageDomainToFourierDomain)

        self.proxy.init_cache(
            self.subgrid_size, self.cell_size, self.w_step, self.shift
        )

        self.aterm_offsets = get_aterm_offsets(
            self.nr_phase_updates, self.ampl_interval
        )

        self.Bampl, self.Tampl = expand_basis_functions(
            self.ampl_poly, self.subgrid_size, self.image_size
        )
        self.Bphase, self.Tphase = expand_basis_functions(
            self.phase_poly, self.subgrid_size, self.image_size
        )


    def process_buffers(self):
        """
        Processing the buffers. This is the central method within any class that
        derives from dp3.Step
        """

        if not self.is_initialized:
            self.initialize()

        # Concatenate accumulated data and display just the shapes
        visibilities = self.__extract_buffer("visibilities")
        flags = self.__extract_buffer("flags")
        weights = self.__extract_buffer("weights")
        uvw_ = self.__extract_buffer("uvw", apply_autocorr_mask=False)
        uvw = np.zeros(shape=(self.nr_baselines, self.ampl_interval, 3), dtype=np.float32)

        uvw[..., 0] = uvw_[self.auto_corr_mask, :, 0]
        uvw[..., 1] = -uvw_[self.auto_corr_mask, :, 1]
        uvw[..., 2] = -uvw_[self.auto_corr_mask, :, 2]

        # Flag NaNs
        flags[np.isnan(visibilities)] = True

        # Set weights of flagged visibilities to zero
        weights *= ~flags

        # Even with weight=0, NaNs still propagate, so set NaN visiblities to zero
        visibilities[np.isnan(visibilities)] = 0.0

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
        X0 = np.ones((self.nr_stations, 1))
        X1 = np.zeros((self.nr_stations, self.ampl_poly.nr_coeffs - 1))
        X2 = np.zeros((self.nr_stations, self.nr_parameters - self.ampl_poly.nr_coeffs))

        parameters = np.concatenate((X0, X1, X2), axis=1)
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

        # parameters has shape (nr_stations, nr_coeffs) for amplitude and (nr_stations, nr_phase_updates, nr_coeffs)
        # for amplitude
        # amplitude/phase basis (Bampl/Bphase) (nr_coeffs, subgridsize, subgridzise, nr_polarizations)
        aterm_ampl = np.tensordot(
            parameters[:, : self.ampl_poly.nr_coeffs], self.Bampl, axes=((1,), (0,))
        )
        aterm_phase = np.exp(
            1j
            * np.tensordot(
                parameters[:, self.ampl_poly.nr_coeffs :].reshape(
                    (self.nr_stations, self.nr_phase_updates, self.phase_poly.nr_coeffs)
                ),
                self.Bphase,
                axes=((2,), (0,)),
            )
        )

        # aterms will have shape (nr_phase_updates, nr_stations, subgrid size, subgrid size, nr polarizations)
        aterms = np.ascontiguousarray(
            (aterm_phase.transpose((1, 0, 2, 3, 4)) * aterm_ampl).astype(
                idg.idgtypes.atermtype
            )
        )

        nr_iterations = 0
        converged = False
        previous_residual = 0.0

        max_dx = 0.0

        timer = -time.time()
        timer0 = 0
        timer1 = 0

        while True:
            nr_iterations += 1
            print(f"iteration nr {nr_iterations} ")

            max_dx = 0.0
            norm_dx = 0.0
            residual_sum = 0.0
            for i in range(self.nr_stations):
                print(f"   Station {i}")
                timer1 -= time.time()

                # Predict visibilities for current solution
                hessian = np.zeros(
                    (self.nr_phase_updates, self.nr_parameters0, self.nr_parameters0),
                    dtype=np.float64,
                )
                gradient = np.zeros(
                    (self.nr_phase_updates, self.nr_parameters0), dtype=np.float64
                )
                residual = np.zeros((1,), dtype=np.float64)

                aterm_ampl = self.__compute_amplitude(i, parameters)
                aterm_phase = self.__compute_phase(i, parameters)
                aterm_derivatives = compute_aterm_derivatives(
                    aterm_ampl, aterm_phase, self.Bampl, self.Bphase
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
                        np.sum(gradient[:, : self.ampl_poly.nr_coeffs], axis=0),
                        gradient[:, self.ampl_poly.nr_coeffs :].flatten(),
                    )
                )

                H00 = hessian[
                    :, : self.ampl_poly.nr_coeffs, : self.ampl_poly.nr_coeffs
                ].sum(axis=0)
                H01 = np.concatenate(
                    [
                        hessian[
                            t, : self.ampl_poly.nr_coeffs, self.ampl_poly.nr_coeffs :
                        ]
                        for t in range(self.nr_phase_updates)
                    ],
                    axis=1,
                )
                H10 = np.concatenate(
                    [
                        hessian[
                            t, self.ampl_poly.nr_coeffs :, : self.ampl_poly.nr_coeffs
                        ]
                        for t in range(self.nr_phase_updates)
                    ],
                    axis=0,
                )
                H11 = scipy.linalg.block_diag(
                    *[
                        hessian[
                            t, self.ampl_poly.nr_coeffs :, self.ampl_poly.nr_coeffs :
                        ]
                        for t in range(self.nr_phase_updates)
                    ]
                )

                hessian = np.block([[H00, H01], [H10, H11]])
                hessian0 = hessian

                dx = np.dot(np.linalg.pinv(hessian, self.pinv_tol), gradient)

                if max_dx < np.amax(abs(dx)):
                    max_dx = np.amax(abs(dx))
                    i_max = i

                parameters[i] += self.solver_update_gain * dx

                # Recompute aterms with updated parameters
                aterm_ampl = self.__compute_amplitude(i, parameters)
                aterm_phase = self.__compute_phase(i, parameters)
                aterms[:, i] = aterm_ampl * aterm_phase

                timer1 += time.time()

            dresidual = previous_residual - residual_sum
            fractional_dresidual = dresidual / residual_sum

            print(max_dx, fractional_dresidual)

            previous_residual = residual_sum

            # converged = (nr_iterations > 1) and (fractional_dresidual < 1e-2)
            converged = (nr_iterations > 1) and (max_dx < 1e-2)

            if converged:
                print(f"Converged after {nr_iterations} iterations - {max_dx}")
                break

            if nr_iterations == self.max_iter:
                print(f"Did not converge after {nr_iterations} iterations - {max_dx}")
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
            :, : self.ampl_poly.nr_coeffs
        ].reshape(self.nr_stations, 1, self.ampl_poly.nr_coeffs)
        # phase parameters: reshaped into (nr_stations, nr_phase_updates, nr_parameters_phase) array
        phase_coefficients = parameters_polynomial[
            :, self.ampl_poly.nr_coeffs : :
        ].reshape(self.nr_stations, self.nr_phase_updates, self.phase_poly.nr_coeffs)

        offset_amplitude = (0, self.count_process_buffer_calls, 0)
        offset_phase = (0, self.count_process_buffer_calls * self.nr_phase_updates, 0)
        self.h5writer.fill_solution_table(
            "amplitude_coefficients", amplitude_coefficients, offset_amplitude
        )
        self.h5writer.fill_solution_table(
            "phase_coefficients", phase_coefficients, offset_phase
        )
        self.count_process_buffer_calls += 1

    def __extract_buffer(self, name, apply_autocorr_mask=True):
        """
        Extract buffer from buffered data.

        Parameters
        ----------
        name : str
            Should be any of ("visibilities", "weights", "flags", "uvw")
        apply_autocorr_mask : bool, optional
            Remove autocorrelation from returned result? Defaults to True

        Returns
        -------
        np.ndarray
        """

        if name == "visibilities":
            result = [
                np.array(dpbuffer.get_data(), copy=False) for dpbuffer in self.dpbuffers
            ]
        elif name == "flags":
            result = [
                np.array(dpbuffer.get_flags(), copy=False)
                for dpbuffer in self.dpbuffers
            ]
        elif name == "weights":
            result = [
                np.array(dpbuffer.get_weights(), copy=False)
                for dpbuffer in self.dpbuffers
            ]
        elif name == "uvw":
            result = [
                np.array(dpbuffer.get_uvw(), copy=False) for dpbuffer in self.dpbuffers
            ]
        else:
            raise ValueError("Name not recognized")
        result = np.stack(result, axis=1)
        return result[self.auto_corr_mask, :, :] if apply_autocorr_mask else result

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
            Array containing the complex exponential, shape is (nr_phase_updates, subgrid_size, subgrid_size, nr_polarizations)
        """
        # Result is repeated nr_phase_updates times to match complex exponential term
        return np.repeat(
            np.tensordot(
                parameters[i, : self.ampl_poly.nr_coeffs],
                self.Bampl,
                axes=((0,), (0,)),
            )[np.newaxis, :],
            self.nr_phase_updates,
            axis=0,
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
            Array containing the complex exponential, shape is (nr_phase_updates, subgrid_size, subgrid_size, nr_polarizations)
        """
        return np.exp(
            1j
            * np.tensordot(
                parameters[i, self.ampl_poly.nr_coeffs :].reshape(
                    (self.nr_phase_updates, self.phase_poly.nr_coeffs)
                ),
                self.Bphase,
                axes=((1,), (0,)),
            )
        )


def compute_aterm_derivatives(aterm_ampl, aterm_phase, B_a, B_p):
    """
    Compute the partial derivatives of g = B_a*x_a * exp(j*B_p*x_p):
    - \partial g / \partial x_a = B_a * exp(j*B_p*x_p)
    - \partial g / \partial x_p = B_a * x_a * j * B_p * exp(j*B_p*x_p)
    where x_a and x_p the unknown expansion coefficients for the amplitude and
    the phase, respectively.

    Parameters
    ----------
    aterm_ampl : np.ndarray
        Amplitude tensor product B_a * x_a, should have shape (nr_phase_updates, subgrid_size, subgrid_size, nr_polarizations)
    aterm_phase : np.ndarray
        Phase tensor product B_p * x_p, should have shape (nr_phase_updates, subgrid_size, subgrid_size, nr_polarizations)
    B_a : np.ndarray
        Expanded (amplitude) basis functions, should have shape (nr_ampl_coeffs, subgrid_size, subgrid_size, nr_polarizations)
    B_p : np.ndarray
        Expande (phase) basis functions, should have shape (nr_phase_coeffs, subgrid_size, subgrid_size, nr_polarizations)

    Returns
    -------
    np.ndarray
        Column stacked derivative [\partial g/ \partial x_a, \partial g / \partial x_p]^T
        Output has shape (nr_phase_updates, nr_coeffs, subgrid_size, subgrid_size, nr_polarizations)
        where nr_coeffs = len(x_a) + len(x_p)
    """
    # new-axis is introduced at "stations" axis
    aterm_derivatives_ampl = (
        aterm_phase[:, np.newaxis, :, :, :] * B_a[np.newaxis, :, :, :, :]
    )

    aterm_derivatives_phase = (
        1j
        * aterm_ampl[:, np.newaxis, :, :, :]
        * aterm_phase[:, np.newaxis, :, :, :]
        * B_p[np.newaxis, :, :, :, :]
    )

    aterm_derivatives = np.concatenate(
        (aterm_derivatives_ampl, aterm_derivatives_phase), axis=1
    )
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
    parameters[:, :nr_amplitude_params] = np.dot(
        parameters[:, :nr_amplitude_params], tmat_amplitude.T
    )

    # Map the phases
    for j in range(nr_timeslots):
        slicer = slice(
            nr_amplitude_params + j * nr_phase_params,
            nr_amplitude_params + (j + 1) * nr_phase_params,
        )
        parameters[:, slicer] = np.dot(parameters[:, slicer], tmat_phase.T)
    return parameters


def expand_basis_functions(polynomial, subgrid_size, image_size):
    """
    Expand the (orthonormalized) Lagrange polynomial basis on a
    given subgrid. Also returns the transformation matrix for the mapping


    Parameters
    ----------
    polynomial : idg.basisfunctions.LagrangePolynomial
        Polynomial to be used in the expansion
    subgrid_size : int
        Size of IDG subgrid (assumed to be square)
    image_size : float
        Size of image, i.e. nr_pixels x cell_size. Image is (assumed to be square)

    Returns
    -------
    np.ndarray, np.ndarray
        np.ndarray with evaluation of orthonormal basis functions and the transformation matrix
        that maps the orthonormal basis onto the "regular" basis
    """
    s = image_size / subgrid_size * (subgrid_size - 1)
    l = s * np.linspace(-0.5, 0.5, subgrid_size)
    m = -s * np.linspace(-0.5, 0.5, subgrid_size)

    basis_functions = polynomial.expand_basis(l, m)

    # Casting dim 3 matrix into dim 2 matrix
    basis_functions = basis_functions.reshape((-1, subgrid_size * subgrid_size)).T

    U, S, V, = np.linalg.svd(basis_functions)
    basis_functions_orthonormal = U[:, : polynomial.nr_coeffs]
    T = np.dot(np.linalg.pinv(basis_functions), basis_functions_orthonormal)
    basis_functions_orthonormal = basis_functions_orthonormal.T.reshape(
        (-1, subgrid_size, subgrid_size, 1)
    )

    # Kronecker product to expand scalar to length 4 vectors
    # representing 2x2 identity (Jones) matrices accounting for polarization
    basis_functions_orthonormal = np.kron(
        basis_functions_orthonormal, np.array([1.0, 0.0, 0.0, 1.0])
    )
    return basis_functions_orthonormal, T
