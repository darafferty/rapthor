import os
import ctypes
import numpy
from ctypes.util import find_library

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

visibilitiestype = numpy.complex64
uvwtype = numpy.dtype([('u', numpy.float32),
                       ('v', numpy.float32),
                       ('w', numpy.float32)])
wavenumberstype = numpy.float32
gridtype = numpy.complex64
baselinetype = numpy.dtype([('station1', numpy.intc),
                            ('station2', numpy.intc)])
coordinatetype = numpy.dtype([('x', numpy.intc),
                              ('y', numpy.intc)])
metadatatype = numpy.dtype([ ('baseline_offset', numpy.intc),
                             ('time_offset', numpy.intc),
                             ('nr_timesteps', numpy.intc),
                             ('aterm_index', numpy.intc),
                             ('baseline', baselinetype),
                             ('coordinate', coordinatetype)])
atermtype = numpy.complex64
atermoffsettype = numpy.intc
spheroidaltype = numpy.float32

FourierDomainToImageDomain = 0
ImageDomainToFourierDomain = 1
