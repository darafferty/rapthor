# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import ctypes
import numpy.ctypeslib
from .Proxy import *

lib = idg.load_library('libidg-opencl.so')

class Generic(Proxy):

    def __init__(
        self,
        nr_correlations,
        subgrid_size):
        """Generic OpenCL implementation"""
        try:
            lib.OpenCL_Generic_init.argtypes = [ctypes.c_uint, \
                                               ctypes.c_uint]
            self.obj = lib.OpenCL_Generic_init(
                ctypes.c_uint(nr_correlations),
                ctypes.c_uint(subgrid_size))
        except AttributeError:
            print("The chosen proxy was not built into the library")