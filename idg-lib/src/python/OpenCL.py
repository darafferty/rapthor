# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

"""
The idg.OpenCL module contains a Proxy subclass that provides access to 
the IDG implementation for devices that support OpenCL.
"""

import os
import ctypes
import numpy.ctypeslib
from .Proxy import *

class OpenCL(Proxy):
    lib = idg.load_library('libidg-opencl.so')

class Generic(OpenCL):
    """
    idg.OpenCL.Generic() creates a Proxy instance that provides access
    to IDG implementation for devices that support OpenCL.
    """

    def __init__(self):
        """Generic OpenCL implementation"""
        try:
            self.lib.OpenCL_Generic_create.restype = ctypes.c_void_p
            self.lib.OpenCL_Generic_create.argtypes = []
            self.obj = self.lib.OpenCL_Generic_create()
        except AttributeError:
            print("The chosen proxy was not built into the library")