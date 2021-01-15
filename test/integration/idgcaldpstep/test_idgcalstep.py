#!/usr/bin/env python3
# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later3

import os
# from astropy.io import fits
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

def run_dppp():
    # Run a IDGCalDPStep with DPPP
    check_call(
        [
            "DPPP",
            os.path.join(WORKDIR, "DPPP.parset")
        ]
    )

def test_idgcalstep():
    run_dppp()
