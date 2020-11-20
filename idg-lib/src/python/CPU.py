# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import ctypes
import numpy.ctypeslib
from .Proxy import *

class CPU(Proxy):
    lib = idg.load_library('libidg-cpu.so')

class Reference(CPU):

    def __init__(
        self,
        nr_correlations,
        subgrid_size):
        """Reference CPU implementation"""
        try:
            self.lib.CPU_Optimized_init.restype = ctypes.c_void_p
            self.lib.CPU_Reference_init.argtypes = [ctypes.c_uint, \
                                               ctypes.c_uint]
            self.obj = self.lib.CPU_Reference_init(
                ctypes.c_uint(nr_correlations),
                ctypes.c_uint(subgrid_size))
        except AttributeError:
            print("The chosen proxy was not built into the library")

class Optimized(CPU):

    def __init__(
        self,
        nr_correlations,
        subgrid_size):
        """Optimized CPU implementation"""
        try:
            self.lib.CPU_Optimized_init.restype = ctypes.c_void_p
            self.lib.CPU_Optimized_init.argtypes = [ctypes.c_uint, \
                                               ctypes.c_uint]
            self.obj = self.lib.CPU_Optimized_init(
                ctypes.c_uint(nr_correlations),
                ctypes.c_uint(subgrid_size))
        except AttributeError:
            print("The chosen proxy was not built into the library")
