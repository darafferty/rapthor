# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

"""
IDG
===

The main class of the IDG module is the Proxy.
The class is named Proxy because its instances act as proxies for the underlying
accelerator hardware.

The IDG library was born out of a project comparing the performance of implementations of gridding/degridding algorithms
on various accelerator architectures, like for example Graphical Processing Units (GPUs).

Currently there are four submodules for different types of proxies: :py:mod:`CPU`, :py:mod:`CUDA`, :py:mod:`OpenCL` and :py:mod:`HybridCUDA`.
Each submodule contains one or more subclasses of the Proxy class.

The two most commonly used proxies are CPU.Optimized and HybridCUDA.GenericOptimized, 
respectively a pure CPU and a CPU+NVIDIA GPU implementation.

A proxy can be instantiated like this, for example::

    import idg
    proxy = idg.HybridCUDA.GenericOptimized()

See the documentation of :py:class:`idg.Proxy.Proxy` on how to use a proxy.


"""

import os
import ctypes
import numpy
from idg.idgtypes import *
import importlib

# A bit ugly, but ctypes.util's find_library does not look in
# the LD_LIBRARY_PATH, but only PATH. Howver, we can also provide
# the full path of the shared object file
def load_library(libname):
    path = os.path.dirname(os.path.realpath(__file__))
    path, _ = os.path.split(path)
    path, _ = os.path.split(path)
    path, _ = os.path.split(path)
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
    except ModuleNotFoundError as e:
        print("Error: %s" % e)
    except OSError as e:
        handle_error(proxy_module, e)


try:
    import idg.fft
    from idg.Plan import *
except OSError as e:
    handle_error("utils", e)
