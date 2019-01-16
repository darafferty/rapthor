import os
import ctypes
import numpy
from ctypes.util import find_library
from idgtypes import *

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
