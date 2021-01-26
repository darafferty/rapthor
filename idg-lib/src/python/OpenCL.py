# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import ctypes
import numpy.ctypeslib
from .Proxy import *

lib = idg.load_library('libidg-opencl.so')

class Generic(Proxy):

    def __init__(self):
        """Generic OpenCL implementation"""
        try:
            self.lib.OpenCL_Generic_create.restype = ctypes.c_void_p
            self.lib.OpenCL_Generic_create.argtypes = []
            self.obj = lib.OpenCL_Generic_create()
        except AttributeError:
            print("The chosen proxy was not built into the library")