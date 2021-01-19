#!/usr/bin/env python3
# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later3

import os
from astropy.io import fits
from subprocess import call, check_call
import numpy as np
# import casacore.tables
# import itertools
# import copy
import h5py
import pytest

# Extract some environment variables
COMMONDIR = os.environ["COMMON"]
DATADIR = os.environ["DATADIR"]
MSNAME = os.environ["MSNAME"]
WORKDIR = os.environ["WORKDIR"]
MS = os.path.join(DATADIR, MSNAME)

CELLSIZE = 2.0  # asecs
IMAGESIZE = 512  # pixels

def run_dppp():
    # Run a IDGCalDPStep with DPPP
    check_call(
        [
            "DPPP",
            os.path.join(WORKDIR, "DPPP.parset")
        ]
    )

def create_input_h5parm():
    pass

def create_model_image():
    """
    Prepare model image 
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

    with fits.open("wsclean-image.fits") as img:

        grid_size = img[0].data.shape[-1]

        N = 8
        step = grid_size // N

        img[0].data[:] = 0.0
        for i in range(N):
            for j in range(N):
                img[0].data[0, 0, step//2 + i*step, step//2 + j*step] = 1.0

        img.writeto("wsclean-model.fits", overwrite=True)

def run_wsclean_predict():
    cmd = (
        "wsclean",
        "-quiet",
        "-predict",
        "-use-idg",
        "-no-reorder",
        MS
    )
    check_call(cmd)

def run_compare_h5parm():
    pass

def test_idgcalstep():
    create_input_h5parm()
    create_model_image()
    run_wsclean_predict()
    run_dppp()
    run_compare_h5parm()
