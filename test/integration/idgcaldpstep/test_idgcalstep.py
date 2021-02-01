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
MSNAME = "LOFAR_MOCK.ms"
WORKDIR = os.environ["WORKDIR"]
MS = os.path.join(DATADIR, MSNAME)

CELLSIZE = 2.0  # asecs
IMAGESIZE = 512  # pixels

def run_dppp():
    # Run a IDGCalDPStep with DPPP
    check_call(
        [
            "DPPP",
            os.path.join(WORKDIR, "DPPP.parset"),
            f"msin={MS}"
        ]
    )

def create_input_h5parm(h5parm_in):
    f = h5py.File(h5parm_in, 'r+')
    amplitudes = f['coefficients000']['amplitude_coefficients']['val']
    phases = f['coefficients000']['phase_coefficients']['val']

    amplitudes[:] = np.random.uniform(low=0.5, high=2.0, size=amplitudes.shape)
    phases[:] = np.random.uniform(low=-np.pi/4, high=np.pi/4, size=phases.shape)
    f.close()

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
        "-aterm-config", "aterm.conf",
        MS
    )
    check_call(cmd)

def run_compare_h5parm(h5parm_in, h5parm_out):
    f_in = h5py.File(h5parm_in)
    f_out = h5py.File(h5parm_out)

    amplitude_values_in = f_in['coefficients000']['amplitude_coefficients']['val'][:]
    amplitude_values_out = f_out['coefficients000']['amplitude_coefficients']['val'][:]
    # phase_values = f['coefficients000']['phase_coefficients']['val'][:]

    print(amplitude_values_in)
    print(amplitude_values_out)
    print(np.abs(amplitude_values_in - amplitude_values_out))
    # print(phase_values)

def test_idgcalstep():
    create_input_h5parm("idgcal_in.h5parm")
    create_model_image()
    run_wsclean_predict()
    run_dppp()
    run_compare_h5parm("idgcal_in.h5parm","idgcal_out.h5parm")
