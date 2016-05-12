import os
import ctypes
import numpy
from ctypes.util import find_library

# A bit ugly, but ctypes.util's find_library does not look in
# the LD_LIBRARY_PATH, but only PATH. Howver, we can also provide
# the full path of the shared object file
path = os.path.dirname(os.path.realpath(__file__))
path, junk = os.path.split(path)
path, junk = os.path.split(path)
libpath = os.path.join(path, 'libidg.so')
lib = ctypes.cdll.LoadLibrary(libpath)


class GridderPlan():
    """Create a Gridder plan"""
    def __init__(self, bufferTimesteps):
        lib.GridderPlan_init.argtypes = [ctypes.c_uint]
        self.obj = lib.GridderPlan_init(
            ctypes.c_uint(bufferTimesteps)
        )
