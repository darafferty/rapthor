# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later3

import sys

sys.path.append("..")
from basisfunctions import LagrangePolynomial
import numpy as np
import pytest


def explicit_evaluation(x, y, order, coeffs):
    sol = 0.0
    idx = 0
    for n in range(order + 1):
        for k in range(n + 1):
            sol += coeffs[idx] * pow(x, n - k) * pow(y, k)
            idx += 1
    return sol


@pytest.mark.parametrize(
    "x, y, coeffs",
    [
        # Scalar input
        (1.0, 2.0, np.array(range(1, 2))),
        (1.0, 2.0, np.array(range(1, 4))),
        (1.0, 2.0, np.array(range(1, 7))),
        (1.0, 2.0, np.array(range(1, 11))),
        # x vector, y scalar (only third order polynomial)
        (np.array(range(-2, 10)), 2.0, np.array(range(1, 11))),
        # x scalar, y vector (only third order polynomial)
        (2.0, np.array(range(-2, 10)), np.array(range(1, 11))),
        # both vector (outcome is 2d matrix)
        (np.array(range(-2, 10)), np.array(range(-2, 10)), np.array(range(1, 11))),
    ],
)
def test_evaluate(x, y, coeffs):
    poly = LagrangePolynomial(nr_coeffs=coeffs.size)

    result = poly.evaluate(x, y, coeffs)

    # Cast ref to a convenient shape
    if not isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
        result = np.asarray(result).reshape(1, 1)
    elif not isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        # cast 1d result to 2d array
        result = result.reshape(y.size, 1)
    elif isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
        # cast 1d result to 2d array
        result = result.reshape(1, x.size)
    # no need to do anything for x, and y vector

    x = np.array([x]) if not isinstance(x, np.ndarray) else x
    y = np.array([y]) if not isinstance(y, np.ndarray) else y

    ref = np.empty((y.size, x.size), dtype=result.dtype)
    for i, ycoord in enumerate(y):
        for j, xcoord in enumerate(x):
            ref[i, j] = explicit_evaluation(xcoord, ycoord, poly.order, coeffs)
    np.testing.assert_equal(result, ref)


@pytest.mark.parametrize(
    "order, x, y",
    [
        (0, 2, 3),
        (1, 2, 3),
        (2, 2, 3),
        (3, 2, 3),
        (1, np.array([1, 2]), np.array([2, 3])),
    ],
)
def test_expand(order, x, y):
    poly = LagrangePolynomial(order=order)

    basis = poly.expand_basis(x, y)
    if basis.ndim == 2:
        i = 0
        ref = []
        while i <= order:
            if i == 0:
                ref.append(np.array([1]))
            if i == 1:
                ref.append(np.array([2, 3]))
            if i == 2:
                ref.append(np.array([4, 6, 9]))
            if i == 3:
                ref.append(np.array([8, 12, 18, 27]))
            i += 1
        ref = np.concatenate(ref)
        np.testing.assert_equal(basis.flatten(), ref)

    if basis.ndim == 3:
        X, Y = np.meshgrid(x, y)
        np.testing.assert_equal(basis[0], np.ones((2, 2)))
        np.testing.assert_equal(basis[1], X)
        np.testing.assert_equal(basis[2], Y)
