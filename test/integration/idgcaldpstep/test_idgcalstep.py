# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later3

import os
from astropy.io import fits
from subprocess import call, check_call
import numpy as np
import casacore.tables
import h5py
import pytest
from idg.h5parmwriter import H5ParmWriter
from idg.basisfunctions import LagrangePolynomial
from idg.idgcalutils import init_h5parm_solution_table


# Extract some environment variables
COMMONDIR = os.environ["COMMON"]
DATADIR = os.environ["DATADIR"]
MSNAME = os.environ["MSNAME"]
WORKDIR = os.environ["WORKDIR"]
MS = os.path.join(DATADIR, MSNAME)

CELLSIZE = 2.0  # asecs
GRIDSIZE = 512  # pixels
AMPLORDER = 0
PHASEORDER = 1
AMPLINTERVAL=150
PHASEINTERVAL=75
TOLERANCE = 1e-2

def run_dppp():
    # Run a IDGCalDPStep with DPPP
    check_call(
        [
            "DPPP",
            os.path.join(COMMONDIR, "dppp-idg-cal.parset"),
            f"msin={MS}",
            f"idgcal.polynomialdegamplitude={AMPLORDER}",
            f"idgcal.polynomialdegphase={PHASEORDER}",
            f"idgcal.solintamplitude={AMPLINTERVAL}",
            f"idgcal.solintphase={PHASEINTERVAL}"
        ]
    )

def create_input_h5parm(h5parm_in):

    ms_table = casacore.tables.table(MS)
    antenna_table = casacore.tables.table(f"{MS}::ANTENNA")
    antenna_names = np.array(antenna_table.getcol("NAME"))
    antenna_positions = antenna_table.getcol("POSITION")

    nr_stations = len(antenna_names)

    time_table = casacore.tables.taql("SELECT UNIQUE TIME FROM $ms_table")

    time_array = time_table.getcol("TIME")

    dt = time_array[1] - time_array[0]

    # Get time centroids per amplitude/phase solution interval
    time_array_ampl = (
        time_array[:: AMPLINTERVAL] + (AMPLINTERVAL - 1) * dt / 2.0
    )
    time_array_phase = (
        time_array[:: PHASEINTERVAL] + (PHASEINTERVAL - 1) * dt / 2.0
    )

    # Initialize amplitude and phase polynomial
    ampl_poly = LagrangePolynomial(order=AMPLORDER)
    phase_poly = LagrangePolynomial(order=PHASEORDER)

    # Axes data
    axes_labels = ["ant", "time", "dir"]
    axes_data_amplitude = dict(
        zip(
            axes_labels,
            (nr_stations, time_array_ampl.size, ampl_poly.nr_coeffs),
        )
    )
    axes_data_phase = dict(
        zip(
            axes_labels,
            (
                nr_stations,
                time_array_phase.size,
                phase_poly.nr_coeffs,
            ),
        )
    )

    # Initialize h5parm file
    h5writer = H5ParmWriter(
        h5parm_in,
        solution_set_name='coefficients000',
        overwrite=True,
    )

    # Add antenna/station info
    h5writer.add_antennas(
        antenna_names, antenna_positions
    )

    init_h5parm_solution_table(
        h5writer,
        "amplitude",
        axes_data_amplitude,
        antenna_names,
        time_array_ampl,
        0.0,
        0,
    )
    init_h5parm_solution_table(
        h5writer,
        "phase",
        axes_data_phase,
        antenna_names,
        time_array_phase,
        0.0,
        0,
    )

    amplitude_coefficients = np.random.uniform(
        low=0.5, 
        high=2.0, size=(nr_stations, time_array_ampl.size, ampl_poly.nr_coeffs))

    image_size = GRIDSIZE * CELLSIZE/3600/180*np.pi
    phase_coefficients = np.zeros((nr_stations, time_array_phase.size, phase_poly.nr_coeffs))
    phase_coefficients[:,:,0] = np.random.uniform(low=-np.pi/4, high=np.pi/4, size=phase_coefficients.shape[:2])
    phase_coefficients[:,:,1] = np.random.uniform(low=-np.pi/4, high=np.pi/4, size=phase_coefficients.shape[:2])/image_size
    phase_coefficients[:,:,2] = np.random.uniform(low=-np.pi/4, high=np.pi/4, size=phase_coefficients.shape[:2])/image_size

    h5writer.fill_solution_table("amplitude_coefficients", amplitude_coefficients)
    h5writer.fill_solution_table("phase_coefficients", phase_coefficients)

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
            f"{GRIDSIZE}",
            f"{GRIDSIZE}",
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
        "-aterm-config", os.path.join(COMMONDIR, "aterm.conf"),
        MS
    )
    check_call(cmd)

def run_compare_h5parm(h5parm_in, h5parm_out):
    f_in = h5py.File(h5parm_in)
    f_out = h5py.File(h5parm_out)

    amplitude_values_in = f_in['coefficients000']['amplitude_coefficients']['val'][:]
    amplitude_values_out = f_out['coefficients000']['amplitude_coefficients']['val'][:]

    ampl_err = np.amax(np.abs(amplitude_values_in - amplitude_values_out))
    assert(ampl_err < TOLERANCE)

    phase_values_in = f_in['coefficients000']['phase_coefficients']['val'][:]
    phase_values_out = f_out['coefficients000']['phase_coefficients']['val'][:]

    phase_values_in -= np.mean(phase_values_in, axis=0, keepdims=True)
    phase_values_out -= np.mean(phase_values_out, axis=0, keepdims=True)

    # Scale linear terms by image_size/2 to obtain maximum value, 
    # wich occurs at the edge of the image
    image_size = GRIDSIZE * CELLSIZE/3600/180*np.pi
    phase_values_in[:,:,1] *= image_size/2
    phase_values_in[:,:,2] *= image_size/2
    phase_values_out[:,:,1] *= image_size/2
    phase_values_out[:,:,2] *= image_size/2

    phase_err = np.amax(np.abs(phase_values_in - phase_values_out))
    assert(phase_err < TOLERANCE)

def test_idgcalstep():
    create_input_h5parm("idgcal_in.h5parm")
    create_model_image()
    run_wsclean_predict()
    run_dppp()
    run_compare_h5parm("idgcal_in.h5parm","idgcal_out.h5parm")
