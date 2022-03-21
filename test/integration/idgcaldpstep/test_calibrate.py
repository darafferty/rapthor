# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import casacore.tables
import idg
import idg.util as util

import pytest
import os

from utils import read_fits_parameters, init_buffers, init_tapered_grid

"""
Script for testing the calibrate_init and the calibrate_update methods.
Please note that this test needs to be ran after installation, since
the idg libraries are pieced together during installation.
"""

MS = os.path.join(os.environ["DATADIR"], os.environ["MSNAME"])
IMAGENAME = os.path.join(os.environ["DATADIR"], os.environ["MODELIMAGE"])

PROXIES = ["idg.CPU.Optimized", "idg.HybridCUDA.GenericOptimized"]


@pytest.fixture(params=PROXIES)
def set_parameters(request):
    params = {}
    params["proxy"] = request.param
    params["nr_timesteps"] = 4
    params["nr_timeslots"] = 2
    params["nr_correlations"] = 4
    # Image related
    params["padding"] = 1.2
    params["subgrid_size"] = 40
    params["taper_support"] = 7
    params["wterm_support"] = 7
    params["aterm_support"] = 11
    # Base function related
    params["nr_ampl_terms"] = 6
    params["nr_phase_terms"] = 3

    params["w_step"] = 400.0
    params["shift"] = np.array((0.0, 0.0), dtype=np.float32)
    yield params


def read_ms(nr_timesteps, nr_correlations):
    ms_dict = {}
    table = casacore.tables.taql(f"SELECT * FROM {MS} WHERE ANTENNA1 != ANTENNA2")

    nr_timesteps_in_ms = len(casacore.tables.taql("SELECT UNIQUE TIME FROM $table"))
    if nr_timesteps > nr_timesteps_in_ms:
        raise ValueError(
            f"Number of requested timesteps larger than available timesteps in MS, {nr_timesteps} > {nr_timesteps_in_ms}"
        )

    t_ant = casacore.tables.table(table.getkeyword("ANTENNA"))
    t_spw = casacore.tables.table(table.getkeyword("SPECTRAL_WINDOW"))

    ms_dict["nr_stations"] = len(t_ant)
    nr_baselines = (ms_dict["nr_stations"] * (ms_dict["nr_stations"] - 1)) // 2
    ms_dict["nr_baselines"] = nr_baselines
    nr_channels = table[0]["DATA"].shape[0]
    ms_dict["nr_channels"] = nr_channels
    ms_dict["frequencies"] = np.asarray(t_spw[0]["CHAN_FREQ"], dtype=np.float32)[
        :nr_channels
    ]

    nr_rows = ms_dict["nr_baselines"] * nr_timesteps
    ms_dict["antenna1"] = table.getcol("ANTENNA1", startrow=0, nrow=nr_rows).reshape(
        nr_timesteps, nr_baselines
    )
    ms_dict["antenna2"] = table.getcol("ANTENNA2", startrow=0, nrow=nr_rows).reshape(
        nr_timesteps, nr_baselines
    )
    ms_dict["uvw"] = table.getcol("UVW", startrow=0, nrow=nr_rows).reshape(
        nr_timesteps, nr_baselines, 3
    )
    ms_dict["visibilities"] = table.getcol("DATA", startrow=0, nrow=nr_rows)[
        :, : ms_dict["nr_channels"], :
    ].reshape(nr_timesteps, nr_baselines, nr_channels, nr_correlations)
    ms_dict["flags"] = table.getcol("FLAG", startrow=0, nrow=nr_rows)[
        :, : ms_dict["nr_channels"], :
    ].reshape(nr_timesteps, nr_baselines, nr_channels, nr_correlations)
    return ms_dict


@pytest.mark.parametrize("params", [pytest.lazy_fixture("set_parameters")])
def test_idgcal(params):
    fits_settings = read_fits_parameters(IMAGENAME)
    N0 = fits_settings["N0"]
    ms_dict = read_ms(params["nr_timesteps"], params["nr_correlations"])

    uvw, visibilities, weights = init_buffers(
        ms_dict["nr_baselines"],
        ms_dict["nr_channels"],
        params["nr_timesteps"],
        params["nr_correlations"],
    )

    # User settings
    nr_timesteps = params["nr_timesteps"]
    nr_timeslots = params["nr_timeslots"]
    nr_correlations = params["nr_correlations"]
    subgrid_size = params["subgrid_size"]

    kernel_size = (
        params["taper_support"] + params["wterm_support"] + params["aterm_support"]
    )

    nr_parameters_ampl = params["nr_ampl_terms"]
    nr_parameters_phase = params["nr_phase_terms"]
    nr_parameters0 = nr_parameters_ampl + nr_parameters_phase
    nr_parameters = nr_parameters_ampl + nr_parameters_phase * nr_timeslots

    # MS related
    nr_stations = ms_dict["nr_stations"]
    nr_baselines = ms_dict["nr_baselines"]
    nr_channels = ms_dict["nr_channels"]
    frequencies = ms_dict["frequencies"]

    # Init proxy
    d = np.zeros(shape=(N0, N0), dtype=np.float32)
    d[range(200, N0 - 200, 200), range(200, N0 - 200, 200)] = 1.0
    taper, grid = init_tapered_grid(
        params["subgrid_size"],
        params["taper_support"],
        params["padding"],
        params["nr_correlations"],
        N0=N0,
        d=d,
    )

    try:
        proxy = eval(params["proxy"])()
    except:
        pytest.skip(f"could not instantiate proxy {params['proxy']}")
    proxy.set_grid(grid)
    proxy.transform(idg.ImageDomainToFourierDomain)

    proxy_ref = idg.CPU.Optimized()
    proxy_ref.set_grid(grid)

    antenna1_block = ms_dict["antenna1"]
    antenna2_block = ms_dict["antenna2"]
    uvw_block = ms_dict["uvw"]
    vis_block = ms_dict["visibilities"]
    flags_block = ms_dict["flags"]
    weight_block = np.ones(flags_block.shape) * ~flags_block

    vis_block[np.isnan(vis_block)] = 0
    uvw_block[:, 1:3] = -uvw_block[:, 1:3]

    # Change precision
    uvw_block = uvw_block.astype(np.float32)

    # Construct baseline array
    baselines = np.array(
        [(a1, a2) for a1, a2 in zip(antenna1_block[0, :], antenna2_block[0, :])],
        dtype=np.int32,
    )

    # Transpose uvw, visibilities and weights
    uvw[:]  = uvw_block.transpose((1, 0, 2))
    visibilities[...] = vis_block.transpose((1, 0, 2, 3))
    weights[...] = weight_block.transpose((1, 0, 2, 3))

    # Grid visibilities
    aterms = util.get_identity_aterms(
        nr_timeslots, nr_stations, subgrid_size, nr_correlations
    )
    aterms_offsets = util.get_example_aterms_offset(nr_timeslots, nr_timesteps)

    B0 = np.ones((1, subgrid_size, subgrid_size, 1))

    x = np.linspace(-0.5, 0.5, subgrid_size)

    B1, B2 = np.meshgrid(x, x)
    B1 = B1[np.newaxis, :, :, np.newaxis]
    B2 = B2[np.newaxis, :, :, np.newaxis]
    B3 = B1 * B1
    B4 = B2 * B2
    B5 = B1 * B2

    BB = np.concatenate((B0, B1, B2, B3, B4, B5))
    B = np.kron(BB, np.array([1.0, 0.0, 0.0, 1.0]))

    Bampl = B[:nr_parameters_ampl]
    Bphase = B[:nr_parameters_phase]

    proxy.init_cache(
        subgrid_size, fits_settings["cell_size"], params["w_step"], params["shift"]
    )
    proxy_ref.init_cache(
        subgrid_size, fits_settings["cell_size"], params["w_step"], params["shift"]
    )

    X0 = np.zeros((nr_stations, 1))
    X1 = np.ones(X0.shape)
    X2 = -0.2 * np.ones(X0.shape) + 0.4 * np.random.random(X0.shape)

    parameters = np.concatenate(
        (X1,)
        + (nr_parameters_ampl - 1) * (X0,)
        + (X2,)
        + (nr_parameters_phase * nr_timeslots - 1) * (X0,),
        axis=1,
    )

    aterm_ampl = np.tensordot(
        parameters[:, :nr_parameters_ampl], Bampl, axes=((1,), (0,))
    )
    aterm_phase = np.exp(
        1j
        * np.tensordot(
            parameters[:, nr_parameters_ampl:].reshape(
                (nr_stations, nr_timeslots, nr_parameters_phase)
            ),
            Bphase,
            axes=((2,), (0,)),
        )
    )
    aterms[:, :, :, :, :] = aterm_phase.transpose((1, 0, 2, 3, 4)) * aterm_ampl

    uvw1 = np.zeros(
        shape=(nr_stations, nr_stations - 1, nr_timesteps, 3), dtype=np.float32
    )
    visibilities1 = np.zeros(
        shape=(
            nr_stations,
            nr_stations - 1,
            nr_timesteps,
            nr_channels,
            nr_correlations,
        ),
        dtype=idg.visibilitiestype,
    )
    weights1 = np.zeros(
        shape=(
            nr_stations,
            nr_stations - 1,
            nr_timesteps,
            nr_channels,
            nr_correlations,
        ),
        dtype=np.float32,
    )
    baselines1 = np.zeros(shape=(nr_stations, nr_stations - 1, 2), dtype=np.int32)

    for bl in range(nr_baselines):
        # Set baselines
        antenna1 = antenna1_block[0, bl]
        antenna2 = antenna2_block[0, bl]

        bl1 = antenna2 - (antenna2 > antenna1)
        baselines1[antenna1][bl1] = (antenna1, antenna2)

        # Set uvw
        uvw1[antenna1][bl1] = uvw[bl]

        # Set visibilities
        visibilities1[antenna1][bl1] = visibilities[bl]
        weights1[antenna1][bl1] = weights[bl]

        antenna1, antenna2 = antenna2, antenna1
        bl1 = antenna2 - (antenna2 > antenna1)
        baselines1[antenna1][bl1] = (antenna1, antenna2)

        # Set uvw
        uvw1[antenna1][bl1] = -uvw[bl]

        # Set visibilities
        visibilities1[antenna1][bl1] = np.conj(
            visibilities[bl, :, :, (0, 2, 1, 3)].transpose((1, 2, 0))
        )
        weights1[antenna1][bl1] = weights[bl, :, :, (0, 2, 1, 3)].transpose((1, 2, 0))

    predicted_visibilities = np.zeros_like(visibilities)
    print(f"Shape predicted visibilities {predicted_visibilities.shape}")
    proxy_ref.degridding(
        kernel_size,
        frequencies,
        predicted_visibilities,
        uvw,
        baselines,
        aterms,
        aterms_offsets,
        taper,
    )
    residual_visibilities = visibilities - predicted_visibilities

    proxy.calibrate_init(
        kernel_size,
        frequencies.reshape((1,-1)),
        visibilities,
        weights,
        uvw,
        baselines,
        aterms_offsets,
        taper,
    )

    for i in range(nr_stations):
        bl_sel = [i in bl for bl in baselines]
        # Predict visibilities for current solution
        hessian = np.zeros(
            (nr_timeslots, nr_parameters0, nr_parameters0), dtype=np.float64
        )
        gradient = np.zeros((nr_timeslots, nr_parameters0), dtype=np.float64)
        residual = np.zeros((1,), dtype=np.float64)

        # TODO: can be condensed once refactor is merged
        aterm_ampl = np.repeat(
            np.tensordot(parameters[i, :nr_parameters_ampl], Bampl, axes=((0,), (0,)))[
                np.newaxis, :
            ],
            nr_timeslots,
            axis=0,
        )
        aterm_phase = np.exp(
            1j
            * np.tensordot(
                parameters[i, nr_parameters_ampl:].reshape(
                    (nr_timeslots, nr_parameters_phase)
                ),
                Bphase,
                axes=((1,), (0,)),
            )
        )

        aterm_derivatives_ampl = (
            aterm_phase[:, np.newaxis, :, :, :] * Bampl[np.newaxis, :, :, :, :]
        )

        aterm_derivatives_phase = (
            1j
            * aterm_ampl[:, np.newaxis, :, :, :]
            * aterm_phase[:, np.newaxis, :, :, :]
            * Bphase[np.newaxis, :, :, :, :]
        )

        aterm_derivatives = np.concatenate(
            (aterm_derivatives_ampl, aterm_derivatives_phase), axis=1
        )
        aterm_derivatives = np.ascontiguousarray(aterm_derivatives, dtype=np.complex64)

        proxy.calibrate_update(
            i, aterms[np.newaxis,...], aterm_derivatives[np.newaxis,...], 
            hessian[np.newaxis,...], gradient[np.newaxis,...], residual
        )

        # Predict visibilities for current solution
        predicted_visibilities1 = np.zeros(
            shape=(
                nr_parameters + 1,
                nr_stations - 1,
                nr_timesteps,
                nr_channels,
                nr_correlations,
            ),
            dtype=idg.visibilitiestype,
        )
        aterms_local = aterms.copy()

        # iterate over degrees of freedom
        for j in range(nr_parameters0 + 1):
            # fill a-term with derivative
            if j > 0:
                aterms_local[:, i, :, :, :] = aterm_derivatives[:, j - 1, :, :, :]

            proxy_ref.degridding(
                kernel_size,
                frequencies,
                predicted_visibilities1[j],
                uvw1[i],
                baselines1[i],
                aterms_local,
                aterms_offsets,
                taper,
            )

        ## compute residual visibilities
        residual_visibilities1 = visibilities1[i] - predicted_visibilities1[0]

        residual_ref1 = np.sum(
            weights1[i] * residual_visibilities1 * np.conj(residual_visibilities1)
        ).real
        residual_ref = np.sum(
            residual_visibilities[bl_sel]
            * residual_visibilities[bl_sel].conj()
            * weights[bl_sel]
        ).real
        # compute vector and  matrix
        v = np.zeros((nr_timeslots, nr_parameters0, 1), dtype=np.complex64)
        M = np.zeros((nr_timeslots, nr_parameters0, nr_parameters0), dtype=np.complex64)

        for l in range(nr_timeslots):
            time_idx = slice(aterms_offsets[l], aterms_offsets[l + 1])
            for j in range(nr_parameters0):
                v[l, j] = np.sum(
                    weights1[i, :, time_idx, :, :]
                    * residual_visibilities1[:, time_idx, :, :]
                    * np.conj(predicted_visibilities1[j + 1, :, time_idx, :, :])
                )
                for k in range(j + 1):
                    M[l, j, k] = np.sum(
                        weights1[i, :, time_idx, :, :]
                        * predicted_visibilities1[j + 1, :, time_idx, :, :]
                        * np.conj(predicted_visibilities1[k + 1, :, time_idx, :, :])
                    )
                    M[l, k, j] = np.conj(M[l, j, k])

        hessian_err = np.amax(
            (abs(hessian) > 1e-3)
            * abs(hessian - np.real(M))
            / np.maximum(abs(hessian), 1.0)
        )
        gradient_err = np.amax(
            abs(gradient - np.real(v[:, :, 0])) / np.maximum(abs(gradient), 1.0)
        )
        residual_err = abs(residual[0] - residual_ref) / residual[0]

        assert (
            hessian_err < 1e-4
        ), f"Hessian error {hessian_err:.2e} for station {i} larger than threshold of 1e-4"
        assert (
            gradient_err < 1e-4
        ), f"Gradient error {gradient_err:.2e} for station {i} larger than threshold of 1e-4"
        assert (
            residual_err < 1e-4
        ), f"Residual error {residual_err:.2e} for station {i} larger than threshold of 1e-4"
