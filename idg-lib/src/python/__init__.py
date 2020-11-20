# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import ctypes
import numpy
from ctypes.util import find_library
from idg.idgtypes import *
import importlib

# A bit ugly, but ctypes.util's find_library does not look in
# the LD_LIBRARY_PATH, but only PATH. Howver, we can also provide
# the full path of the shared object file
def load_library(libname):
    path = os.path.dirname(os.path.realpath(__file__))
    path, junk = os.path.split(path)
    path, junk = os.path.split(path)
    path, junk = os.path.split(path)
    libpath = os.path.join(path, libname)
    lib = ctypes.cdll.LoadLibrary(libpath)
    return lib

def handle_error(library, e):
    if "libidg" in e.args:
        # cannot load idg library (probably because it is not build)
        pass
    else:
        print("Error importing %s: %s" % (library, e.args))

for proxy_module in ["CPU", "CUDA", "OpenCL", "HybridCUDA"]:
    try:
        globals()[proxy_module] = importlib.import_module("." + proxy_module, __name__)
    except OSError as e:
        handle_error(proxy_module, e)


try:
    import idg.fft
    from idg.Plan import *
except OSError as e:
    handle_error("utils", e)
