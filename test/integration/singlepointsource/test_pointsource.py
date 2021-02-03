#!/usr/bin/env python3
# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later3

import os
from astropy.io import fits
from subprocess import call, check_call
import numpy as np
import casacore.tables
import itertools
import copy
import pytest

# Extract some environment variables
COMMONDIR = os.environ["COMMON"]
DATADIR = os.environ["DATADIR"]
MSNAME = os.environ["MSNAME"]
MS = os.path.join(DATADIR, MSNAME)


def test_preparetestset():
    stokes = ["I", "Q", "U", "V"]

    # Make a dummy image using wsclean
    check_call(
        [
            "wsclean",
            "-size",
            "1000",
            "1000",
            "-scale",
            "1asec",
            "-interval",
            "0",
            "1",
            "-no-reorder",
            MS,
        ]
    )

    # Write to a template
    template = os.path.join(DATADIR, "template.fits")
    check_call(["mv", "wsclean-image.fits", template])

    with fits.open(template) as img:
        N = img[0].data.shape[-1]
        # Set all data to 0
        img[0].data[:] = 0.0

        # Set up a filter combining all stokes components, except the combination with itself (i.e. (I, I), (Q, Q), ...)
        stokes_filter = filter(
            lambda pairing: pairing[0] != pairing[1], itertools.product(stokes, stokes)
        )
        # Write
        for (component1, component2) in stokes_filter:
            img.writeto(
                f"pointsource-{component1}-{component2}-model.fits", overwrite=True,
            )

        # Set brightness in center pixel to 1.0, and write the model images for (I, I), (Q, Q) etc:
        img[0].data[0, 0, int(N / 2), int(N / 2)] = 1.0
        for component in stokes:
            img.writeto(
                f"pointsource-{component}-{component}-model.fits", overwrite=True
            )

    # Convert the non-zero (center) pixel to ra-dec coordinates and write cat file
    check_call(["casapy2bbs.py", "pointsource-I-I-model.fits", "pointsource.cat"])

    # Read the input cat file
    file_in = "pointsource.cat"
    with open(file_in, "r") as f_in:
        contents = f_in.readlines()

    # Extract and modify line 8
    patch_info = contents[7].split(",")
    for i, component in enumerate(stokes):
        patch_info_stokes = copy.deepcopy(patch_info)
        if component != "I":
            patch_info_stokes[5] = "0.0"
            patch_info_stokes[5 + i] = patch_info[5]
        contents[7] = ",".join(patch_info_stokes)

        # Write cat files
        file_out = f"pointsource-{component}.cat"
        with open(file_out, "w") as f_out:
            f_out.writelines(contents)

        # And sourcedb files
        sourcedb = f"pointsource-{component}.sourcedb"
        check_call(["rm", "-rf", sourcedb])
        check_call(
            [
                "makesourcedb",
                f"in={file_out}",
                "format=Name, Type, Patch, Ra, Dec, I, Q, U, V",
                f"out={sourcedb}",
            ]
        )


@pytest.mark.parametrize("stokes", ["I", "Q", "U", "V"])
def test_singlepointsource(stokes):
    sourcedb = f"pointsource-{stokes}.sourcedb"
    ms = MS
    T = casacore.tables.taql(
        "SELECT TIME, cdatetime(TIME-.1) AS TIMESTR FROM $ms GROUPBY TIME"
    )

    interval_start = 0
    interval_end = 100

    assert interval_end < len(
        T
    ), f"Interval end set to {interval_end}, but test measurment set only contains {len(T)} timesteps"

    starttime = T[interval_start]["TIME"] - 0.1
    endtime = T[interval_end]["TIME"] - 0.1
    starttimestr = T[interval_start]["TIMESTR"]
    endtimestr = T[interval_end]["TIMESTR"]

    check_call(
        [
            "DPPP",
            os.path.join(COMMONDIR, "dppp-predict.parset"),
            f"msin={MS}",
            f"msin.starttime={starttimestr}",
            f"msin.endtime={endtimestr}",
            f"predict.sourcedb={sourcedb}",
        ]
    )

    check_call(
        [
            "wsclean",
            "-name",
            f"pointsource-{stokes}",
            "-predict",
            "-interval",
            "0",
            "100",
            "-pol",
            "IQUV",
            "-use-idg",
            "-idg-mode",
            "cpu",
            "-no-reorder",
            MS,
        ]
    )

    t = casacore.tables.taql(
        "SELECT * FROM $ms WHERE TIME>$starttime AND TIME<$endtime"
    )

    data = t.getcol("DATA")  # generated by DPPP
    model_data = t.getcol("MODEL_DATA")  # generated by wsclean

    print(f"Norm of DATA column: {np.linalg.norm(data)}")
    print(f"Norm of MODEL_DATA column: {np.linalg.norm(model_data)}")

    err = np.amax(abs(model_data - data))
    assert err < 1.5e-4, f"max error {err} (inf norm) exceeds specified threshold of 1e-4"