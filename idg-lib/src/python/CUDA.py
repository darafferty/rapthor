# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import ctypes
import numpy.ctypeslib
from .Proxy import *

class CUDA(Proxy):
    lib = idg.load_library('libidg-cuda.so')

class Generic(CUDA):

    def __init__(self):
        """Generic CUDA implementation"""
        try:
            self.lib.CUDA_Generic_create.restype = ctypes.c_void_p
            self.lib.CUDA_Generic_create.argtypes = []
            self.obj = self.lib.CUDA_Generic_create()
        except AttributeError:
            print("The chosen proxy was not built into the library")

class Unified(CUDA):

    def __init__(self):
        """Unified CUDA implementation"""
        try:
            self.lib.CUDA_Unified_create.restype = ctypes.c_void_p
            self.lib.CUDA_Unified_create.argtypes = []
            self.obj = self.lib.CUDA_Unified_create()
        except AttributeError:
            print("The chosen proxy was not built into the library")