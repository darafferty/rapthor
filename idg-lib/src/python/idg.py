import os
import ctypes
import numpy
from ctypes.util import find_library
from idgtypes import *

def handle_error(library, e):
    if "libidg" in e.args:
        # cannot load idg library (probably because it is not build)
        pass
    else:
        print("Error importing %s: %s" % (library, e.args))

try:
    import Python
except OSError as e:
    handle_error("Python", e)

try:
    import CPU
except OSError as e:
    handle_error("CPU", e)

try:
    import CUDA
except OSError as e:
    handle_error("CUDA", e)

try:
    import OpenCL
except OSError as e:
    handle_error("OpenCL", e)

try:
    import fft
    from Plan import *
except OSError as e:
    handle_error("utils", e)

try:
    import HybridCUDA
except OSError as e:
    handle_error("HybridCUDA", e)
