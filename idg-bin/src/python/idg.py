import os
import ctypes
import numpy
from ctypes.util import find_library
from idgtypes import *

def handle_error(library, e):
    if "libidg" in e.message:
        # cannot load idg library (probably because it is not build
        pass
    else:
        print("Error importing %s: %s" % (library, e.message))

try:
    import CPU
except OSError as e:
    handle_error("CPU", e)

try:
    import HybridCUDA
except OSError as e:
    handle_error("Hybrid CUDA", e)

try:
    import HybridOpenCL
except OSError as e:
    handle_error("Hybrid OpenCL", e)

try:
    import KNC
except OSError as e:
    handle_error("KNC", e)

try:
    import CUDA
except OSError as e:
    handle_error("CUDA", e)

try:
    import OpenCL
except OSError as e:
    handle_error("OpenCL", e)

try:
    import utils
except OSError:
    handle_error("utils", e)

