# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import ctypes
import numpy.ctypeslib
from .Proxy import *


class HybridCUDA(Proxy):
    lib = idg.load_library('libidg-hybrid-cuda.so')

class GenericOptimized(HybridCUDA):

    def __init__(self):
        """GenericOptimized CUDA implementation"""
        try:
            self.lib.HybridCUDA_GenericOptimized_create.restype = ctypes.c_void_p
            self.lib.HybridCUDA_GenericOptimized_create.argtypes = []
            self.obj = self.lib.HybridCUDA_GenericOptimized_create()
        except AttributeError:
            print("The chosen proxy was not built into the library")