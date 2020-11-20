# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import ctypes
import numpy.ctypeslib
from .Proxy import *

class CUDA(Proxy):
    lib = idg.load_library('libidg-cuda.so')

class Generic(CUDA):

    def __init__(
        self,
        nr_correlations,
        subgrid_size):
        """Generic CUDA implementation"""
        try:
            self.lib.CUDA_Generic_init.restype = ctypes.c_void_p
            self.lib.CUDA_Generic_init.argtypes = [ctypes.c_uint, \
                                               ctypes.c_uint]
            self.obj = self.lib.CUDA_Generic_init(
                ctypes.c_uint(nr_correlations),
                ctypes.c_uint(subgrid_size))
        except AttributeError:
            print("The chosen proxy was not built into the library")

class Unified(CUDA):

    def __init__(
        self,
        nr_correlations,
        subgrid_size):
        """Unified CUDA implementation"""
        try:
            self.lib.CUDA_Unified_init.restype = ctypes.c_void_p
            self.lib.CUDA_Unified_init.argtypes = [ctypes.c_uint, \
                                               ctypes.c_uint]
            self.obj = self.lib.CUDA_Unified_init(
                ctypes.c_uint(nr_correlations),
                ctypes.c_uint(subgrid_size))
        except AttributeError:
            print("The chosen proxy was not built into the library")