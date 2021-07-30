# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

"""
The idg.CUDA module contains Proxy subclasses that provide access to 
IDG implementations for NVIDIA GPUs.
"""

import os
import ctypes
import numpy.ctypeslib
from .Proxy import *

class CUDA(Proxy):
    lib = idg.load_library('libidg-cuda.so')

class Generic(CUDA):
    """
    idg.CUDA.Generic() creates a Proxy instance that provides access
    to the implementation for generic NVIDIA GPU devices.
    """

    def __init__(self):
        """Generic CUDA implementation"""
        try:
            self.lib.CUDA_Generic_create.restype = ctypes.c_void_p
            self.lib.CUDA_Generic_create.argtypes = []
            self.obj = self.lib.CUDA_Generic_create()
        except AttributeError:
            print("The chosen proxy was not built into the library")

class Unified(CUDA):
    """
    idg.CUDA.Generic() creates a Proxy instance that provides access
    to the implementation for NVIDIA GPU devices that support unified memory.
    """

    def __init__(self):
        """Unified CUDA implementation"""
        try:
            self.lib.CUDA_Unified_create.restype = ctypes.c_void_p
            self.lib.CUDA_Unified_create.argtypes = []
            self.obj = self.lib.CUDA_Unified_create()
        except AttributeError:
            print("The chosen proxy was not built into the library")