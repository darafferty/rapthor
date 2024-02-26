# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

from idg.idgtypes import atermoffsettype
import numpy as np
from datetime import datetime

"""
Script containing some utility functions for the idgcaldpstep class
"""


def apply_beam(beam, aterms):
    """
    Apply a beam (Jones matrices) to aterms

    Parameters
    ----------
    beam : np.ndarray
        shape should be ...,2,2
    aterms: np.ndarray
        shape should be ...,4

    Returns
    -------
    Array of the same shape as aterms, matrix multiplied by beam.
    """
    if beam is not None:
        # reshape last axis (4,) to two axes (2,2)
        aterms = np.reshape(aterms, aterms.shape[:-1] + (2, 2))

        # Use Einstein summation to compute the matrix product over the last two axes
        aterms[:] = np.einsum("...ij,...jk->...ik", beam, aterms)

        # reshape two last axes (2,2) to one axis (4,)
        aterms = np.reshape(aterms, aterms.shape[:-2] + (4,))

    return aterms


def expand_basis_functions(polynomial, subgrid_size, image_size, d1=1.0, d2=1.0):
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

    (
        U,
        S,
        V,
    ) = np.linalg.svd(basis_functions)
    basis_functions_orthonormal = U[:, : polynomial.nr_coeffs]
    T = np.dot(np.linalg.pinv(basis_functions), basis_functions_orthonormal)
    basis_functions_orthonormal = basis_functions_orthonormal.T.reshape(
        (-1, subgrid_size, subgrid_size, 1)
    )

    # Kronecker product to expand scalar to length 4 vectors
    # representing 2x2 identity (Jones) matrices accounting for polarization
    basis_functions_orthonormal = np.kron(
        basis_functions_orthonormal, np.array([d1, 0.0, 0.0, d2])
    )
    return basis_functions_orthonormal, T


def next_composite(n):
    """
    Find smallest integer equal to or higher than n that is a composite of prime factors 2,3 and 5.
    This can be used to find a grid size of at least n, for which the FFT can be computed efficiently.

    Parameters
    ----------
    n : int

    Returns
    -------
    int
        smallest composite of prime factors 2,3 and 5, equal to or larger than n.
    """
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
    TODO: Compute an optimal taper of size N, and support W

    Parameters
    ----------
    N : int
        size of taper in pixels
    W : int
        support in the Fourier domain (pixels)
    padding : float
        padding factor used for main grid.
        taper will be optimized for unpadded region,
    offset : float, optional
        For even taper sizes N, an offset of 0.5 (the default) results in a symmetric taper.
    l_range : np.array(dtype=float), optional
        custom sampling of taper.
        For default value None, the taper is sampled at N equidistant points

    Returns
    -------
    np.array(shape=(N,), dtype=np.float)
        Optimal taper
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
        taper = B @ a

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


def get_aterm_offsets(nr_timeslots, nr_time):
    aterm_offsets = np.zeros((nr_timeslots + 1), dtype=atermoffsettype)

    for i in range(nr_timeslots + 1):
        aterm_offsets[i] = i * (nr_time // nr_timeslots)

    return aterm_offsets


def init_h5parm_solution_table(
    h5parm_object,
    soltab_type,
    axes_info,
    antenna_names,
    time_array,
    freq_array,
    image_size,
    subgrid_size,
    basisfunction_type="lagrange",
    history="",
):
    """
    Initialize h5parm solution table

    Parameters
    ----------
    h5parm_object : idg.h5parmwriter.H5ParmWriter
        h5parm object
    soltab_type : str
        Any of ("amplitude", "phase")
    axes_info : dict
        Dict containing axes info (name, length)
    antenna_names : np.ndarray
        Array of strings containing antenna names
    time_array : np.ndarray
        Array of times
    freq_array : np.ndarray
        Array of frequencies
    image_size : float
        Pixel size
    subgrid_size : int
        Subgrid size, used in IDG
    basisfunction_type : str, optional
        Which basis function was used? Defaults to "lagrange"
    history : str, optional
        History attribute, by default ""

    Returns
    -------
    idg.h5parmwriter.H5ParmWriter
        Extended H5ParmWriter object
    """
    soltab_info = {
        "amplitude": "amplitude_coefficients",
        "amplitude1": "amplitude1_coefficients",
        "amplitude2": "amplitude2_coefficients",
        "slowphase1": "slowphase1_coefficients",
        "slowphase2": "slowphase2_coefficients",
        "phase": "phase_coefficients",
    }

    assert soltab_type in soltab_info.keys()
    soltab_name = soltab_info[soltab_type]

    h5parm_object.create_solution_table(
        soltab_name,
        soltab_type,
        axes_info,
        dtype=np.float_,
        history=f'CREATED at {datetime.today().strftime("%Y/%m/%d")}; {history}',
    )

    # Set info for the "ant" axis
    h5parm_object.create_axis_meta_data(soltab_name, "ant", meta_data=antenna_names)

    # Set info for the "dir" axis
    h5parm_object.create_axis_meta_data(
        soltab_name,
        "dir",
        attributes={
            "basisfunction_type": basisfunction_type,
            "image_size": image_size,
            "subgrid_size": subgrid_size,
        },
    )

    # Set info for the "time" axis
    h5parm_object.create_axis_meta_data(soltab_name, "time", meta_data=time_array)

    # Set info for the "freq" axis
    h5parm_object.create_axis_meta_data(soltab_name, "freq", meta_data=freq_array)
