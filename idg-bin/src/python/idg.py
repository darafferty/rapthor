import os
import ctypes
import numpy
from ctypes.util import find_library

try:
    import CPU
except OSError:
    pass

try:
    import Hybrid
except OSError:
    pass

try:
    import KNC
except OSError:
    pass

try:
    import OpenCL
except OSError:
    pass

try:
    import utils
except OSError:
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
