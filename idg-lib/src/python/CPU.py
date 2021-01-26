# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import ctypes
import numpy.ctypeslib
from .Proxy import *

class CPU(Proxy):
    lib = idg.load_library('libidg-cpu.so')

class Reference(CPU):

    def __init__(self):
        """Reference CPU implementation"""
        try:
            self.lib.CPU_Reference_create.restype = ctypes.c_void_p
            self.lib.CPU_Reference_create.argtypes = []
            self.obj = self.lib.CPU_Reference_create()
        except AttributeError:
            print("The chosen proxy was not built into the library")

class Optimized(CPU):

    def __init__(self):
        """Optimized CPU implementation"""
        try:
            self.lib.CPU_Optimized_create.restype = ctypes.c_void_p
            self.lib.CPU_Optimized_create.argtypes = []
            self.obj = self.lib.CPU_Optimized_create()
        except AttributeError:
            print("The chosen proxy was not built into the library")
