import os
import ctypes
import numpy
from ctypes.util import find_library

try:
    import CPU
except OSError as e:
    print("Error importing CPU: ", e)
    pass

try:
    import HybridCUDA
except OSError as e:
    print("Error importing Hybrid CUDA: ", e)
    pass

try:
    import HybridOpenCL
except OSError as e:
    print("Error importing Hybrid OpenCL: ", e)
    pass

try:
    import KNC
except OSError as e:
    print("Error importing KNC: ", e)
    pass

try:
    import OpenCL
except OSError as e:
    print("Error importing OpenCL: ", e)
    pass

try:
    import utils
except OSError:
    print("Error importing utils: ", e)
    pass

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
