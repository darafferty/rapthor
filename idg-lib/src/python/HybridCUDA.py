# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import ctypes
import numpy.ctypeslib
from .Proxy import *


class HybridCUDA(Proxy):
    lib = idg.load_library('libidg-hybrid-cuda.so')

class GenericOptimized(HybridCUDA):

    def __init__(
        self,
        nr_correlations,
        subgrid_size):
        """GenericOptimized CUDA implementation"""
        try:
            self.lib.HybridCUDA_GenericOptimized_init.restype = ctypes.c_void_p
            self.lib.HybridCUDA_GenericOptimized_init.argtypes = [ctypes.c_uint, \
                                               ctypes.c_uint]
            self.obj = self.lib.HybridCUDA_GenericOptimized_init(
                ctypes.c_uint(nr_correlations),
                ctypes.c_uint(subgrid_size))
        except AttributeError:
            print("The chosen proxy was not built into the library")