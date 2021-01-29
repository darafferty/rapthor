# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

from idg.idgtypes import atermoffsettype
import numpy as np

"""
Script containing some utility functions for the idgcaldpstep class
"""


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


def get_aterm_offsets(nr_timeslots, nr_time):
    aterm_offsets = np.zeros((nr_timeslots + 1), dtype=atermoffsettype)

    for i in range(nr_timeslots + 1):
        aterm_offsets[i] = i * (nr_time // nr_timeslots)

    return aterm_offsets
