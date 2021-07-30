# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

"""
The idg.CPU module contains Proxy subclasses that provide access to 
IDG implementations for CPUs.
"""

import os
import ctypes
import numpy.ctypeslib
from .Proxy import *

class CPU(Proxy):
    lib = idg.load_library('libidg-cpu.so')

class Reference(CPU):
    """
    idg.CPU.Reference() creates a Proxy instance that provides access
    to the reference CPU implementation.

    This implementation is not optimized for speed.
    Instead the the underlying C++ code is kept as simple as possible,
    and is intended as reference for other (optimized) implementations.
    """

    def __init__(self):
        """Reference CPU implementation"""
        try:
            self.lib.CPU_Reference_create.restype = ctypes.c_void_p
            self.lib.CPU_Reference_create.argtypes = []
            self.obj = self.lib.CPU_Reference_create()
        except AttributeError:
            print("The chosen proxy was not built into the library")

class Optimized(CPU):
    """
    idg.CPU.Optimized() creates a Proxy instance that provides access
    to a CPU implementation that is optimized for speed.
    """

    def __init__(self):
        """Optimized CPU implementation"""
        try:
            self.lib.CPU_Optimized_create.restype = ctypes.c_void_p
            self.lib.CPU_Optimized_create.argtypes = []
            self.obj = self.lib.CPU_Optimized_create()
        except AttributeError:
            print("The chosen proxy was not built into the library")
