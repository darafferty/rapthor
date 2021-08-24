#!/usr/bin/env python3
# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later3
#
# Gridding/Degridding integration tests. Integration testing of gridding/degridding with
# wsclean-idg. The degridding test compares DPPP generated "DATA" column against wsclean
# generated "MODEL_DATA" column. The gridding tests checks whether the specified flux of the source
# is accurately reconstructed.
#
# This test was extracted from test-main.py in the https://gitlab.com/astron-idg/idg-test-wsclean
# repository.
#
# Please note the following:
#   - To run this test, idg needs to be compiled with -DWRITE_OUT_SCALAR_BEAM=ON in order to write the scalar_beam.npy file.
#   - To limit the computational runtime, this test only checks the Stokes I component. We may want to re-enable the
#     other Stokes components in the future by extending STOKESLIST.
#   - To limit the computational runtime, this test only checks for the more complex case "grid_with_beam=True" and
#     "differential_beam=True"
#   - Currently, this test only runs on the cpu. However, once IDG is compiled with GPU enabled, the ["cpu"] list in load_tests
#     can be changed to ["cpu", "hybrid"] to test on different hardware configurations.
#   - The python tests were inactive in test-main.py, and are therefore not included in this integration test.
#     We may want to re-enable these test cases.
#   - The same holds for "createTestClean". We may want to re-enable this in the future.
#   - As a future improvement, we might consider to eliminate the overlap between preparetestset.py/test_pointsource.py and
#     the test_gridding.py

import pytest
import os
from subprocess import check_call

import numpy as np
from astropy.io import fits
import casacore.tables
from utils import preparetestset

# Extract some environment variables
DATADIR = os.environ["DATADIR"]
COMMONDIR = os.environ["COMMON"]
MSNAME = os.environ["MSNAME"]
MS = os.path.join(DATADIR, MSNAME)

INTERVALSTART = 0
INTERVALEND = 100
STARTCHAN = 0
NCHAN = 0

CELLSIZE = 2.0  # asecs
IMAGESIZE = 512  # pixels

# Only Stokes Q, we may want to re-enable the other components in the future
# STOKESLIST = ["I","Q", "U", "V"]
STOKESLIST = ["Q"]


@pytest.fixture(params=[-128])
def define_nx(request):
    """
    Define the offset in x pixels for the point source
    """
    return request.param


@pytest.fixture(params=[-128])
def define_ny(request):
    """
    Define the offset in y pixels for the point source
    """
    return request.param


@pytest.fixture(params=[False, True])
def define_grid_with_beam(request):
    """
    Use a term correction for beam?
    """
    return request.param


@pytest.fixture(params=[True])
def define_differential_beam(request):
    """
    Use differential beam?
    """
    return request.param


@pytest.fixture(params=["cpu"])
def define_idgmode(request):
    """
    Which idg mode? "cpu" or "hybrid"?
    """
    return request.param


@pytest.fixture(params=STOKESLIST)
def define_stokes(request):
    """
    Which Stokes component? Taken from STOKESLIST global variable
    """
    return request.param


@pytest.mark.parametrize("stokes", [pytest.lazy_fixture("define_stokes")])
def test_setup(stokes):
    """
    Prepare template image and work-around to create pointsource-I-beam-I.fits
    """

    # Make a template of the image
    check_call(
        [
            "wsclean",
            "-quiet",
            "-size",
            f"{IMAGESIZE}",
            f"{IMAGESIZE}",
            "-scale",
            f"{CELLSIZE}asec",
            "-interval",
            "0",
            "1",
            "-no-reorder",
            MS,
        ]
    )
    template = os.path.join(DATADIR, "template.fits")
    check_call(["mv", "wsclean-image.fits", template])

    # Work-around to create a pointsource-I-beam-I.fits in workdir,
    # needed by wsclean later on
    cmd = [
        "wsclean",
        "-quiet",
        "-name",
        f"pointsource-{stokes}",
        "-size",
        f"{IMAGESIZE}",
        f"{IMAGESIZE}",
        "-scale",
        f"{CELLSIZE}asec",
        "-interval",
        f"{INTERVALSTART}",
        f"{INTERVALEND}",
        "-no-reorder",
        "-use-idg",
        "-idg-mode",
        "cpu",
        "-no-dirty",
        "-niter",
        "1",
        "-grid-with-beam",
        MS,
    ]
    check_call(cmd)


@pytest.mark.parametrize(
    "stokes, nx, ny, grid_with_beam, differential_beam, idgmode",
    [
        (
            pytest.lazy_fixture("define_stokes"),
            pytest.lazy_fixture("define_nx"),
            pytest.lazy_fixture("define_ny"),
            pytest.lazy_fixture("define_grid_with_beam"),
            pytest.lazy_fixture("define_differential_beam"),
            pytest.lazy_fixture("define_idgmode"),
        )
    ],
)
def test_degridding(stokes, nx, ny, grid_with_beam, differential_beam, idgmode):
    # Write sourcedb and run DPPP predict
    ms = MS
    preparetestset(
        stokes,
        nx,
        ny,
        grid_with_beam,
        differential_beam,
        DATADIR,
        COMMONDIR,
        ms,
        INTERVALSTART,
        INTERVALEND,
        STARTCHAN,
        NCHAN,
    )

    T = casacore.tables.taql(
        f"SELECT TIME, cdatetime(TIME-.1) AS TIMESTR FROM {ms} GROUPBY TIME"
    )
    starttime = T[INTERVALSTART]["TIME"] - 0.1
    endtime = T[INTERVALEND]["TIME"] - 0.1

    cmd = (
        [
            "wsclean",
            "-quiet",
            "-name",
            f"pointsource-{stokes}",
            "-predict",
            "-interval",
            f"{INTERVALSTART}",
            f"{INTERVALEND}",
            "-pol",
            "IQUV",
            "-use-idg",
            "-idg-mode",
            idgmode,
            "-no-reorder",
        ]
        + (
            ["-channel-range", str(STARTCHAN), str(STARTCHAN + NCHAN)]
            if NCHAN > 0
            else []
        )
        + ["-beam-aterm-update", "30"]
        + (["-grid-with-beam"] if grid_with_beam else [])
        + (
            ["-use-differential-lofar-beam"]
            if (grid_with_beam and differential_beam)
            else []
        )
        + [ms]
    )
    check_call(cmd)

    t = casacore.tables.taql(
        f"SELECT * FROM {ms} WHERE TIME>{starttime} AND TIME<{endtime}  AND ANTENNA1!=ANTENNA2"
    )

    data = t.getcol("DATA")  # generated by DPPP
    if NCHAN > 0:
        data = data[:, STARTCHAN : STARTCHAN + NCHAN, :]

    model_data = t.getcol("MODEL_DATA")  # generated by wsclean
    if NCHAN > 0:
        model_data = model_data[:, STARTCHAN : STARTCHAN + NCHAN, :]

    print(f"Infinity norm data-model_data {np.amax(abs(data-model_data))}")
    if grid_with_beam:
        assert np.allclose(data, model_data, rtol=5e-2, atol=5e-2) == True
    else:
        assert np.allclose(data, model_data, rtol=1e-3, atol=1e-3) == True


@pytest.mark.parametrize(
    "stokes, nx, ny, grid_with_beam, differential_beam, idgmode",
    [
        (
            pytest.lazy_fixture("define_stokes"),
            pytest.lazy_fixture("define_nx"),
            pytest.lazy_fixture("define_ny"),
            pytest.lazy_fixture("define_grid_with_beam"),
            pytest.lazy_fixture("define_differential_beam"),
            pytest.lazy_fixture("define_idgmode"),
        )
    ],
)
def test_gridding(stokes, nx, ny, grid_with_beam, differential_beam, idgmode):
    ms = MS
    preparetestset(
        stokes,
        nx,
        ny,
        grid_with_beam,
        differential_beam,
        DATADIR,
        COMMONDIR,
        ms,
        INTERVALSTART,
        INTERVALEND,
        STARTCHAN,
        NCHAN,
    )

    offset = (int(ny), int(nx))

    name = f"pointsource-{stokes}" + ("-beam" if grid_with_beam else "")

    cmd = (
        [
            "wsclean",
            "-quiet",
            "-name",
            name,
            "-data-column",
            "DATA",
            "-size",
            f"{IMAGESIZE}",
            f"{IMAGESIZE}",
            "-scale",
            f"{CELLSIZE}asec",
            "-interval",
            f"{INTERVALSTART}",
            f"{INTERVALEND}",
            "-no-reorder",
            "-pol",
            "IQUV",
            "-use-idg",
            "-idg-mode",
            idgmode,
            "-no-dirty",
            "-weight",
            "natural",
        ]
        + (
            ["-channel-range", str(STARTCHAN), str(STARTCHAN + NCHAN)]
            if NCHAN > 0
            else []
        )
        + (["-grid-with-beam"] if grid_with_beam else [])
        + (
            ["-use-differential-lofar-beam"]
            if (grid_with_beam and differential_beam)
            else []
        )
        + [ms]
    )
    check_call(cmd)

    if grid_with_beam:
        beam = np.load("scalar_beam.npy")

    for stokes1 in STOKESLIST:
        imgname = f'{name}-{stokes1}-image{["", "-pb"][grid_with_beam]}.fits'
        with fits.open(imgname) as img:
            N = img[0].data.shape[-1]
            flux = img[0].data[0, 0, int(N / 2 + offset[0]), int(N / 2 + offset[1])]

            if grid_with_beam:
                flux /= beam[int(N / 2 + offset[0]), int(N / 2 + offset[1])]

            if (stokes1 == "I") or (stokes == stokes1):
                expected_flux = 1.0
            else:
                expected_flux = 0.0
            assert (
                np.allclose(flux, expected_flux, rtol=1e-2, atol=1e-2) == True
            ), f"Expected flux {stokes1} in Stokes {expected_flux}, found {flux}. Grid with beam? {grid_with_beam}"
