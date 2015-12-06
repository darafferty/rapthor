import os
import ctypes
import numpy
from ctypes.util import find_library

visibilitiestype = numpy.complex64
uvwtype = numpy.float32
wavenumberstype = numpy.float32
gridtype = numpy.complex64
baselinetype = numpy.dtype([('station1', numpy.int32), \
                            ('station2', numpy.int32)])
coordinatetype = numpy.dtype([('x', numpy.int32), \
                              ('y', numpy.int32)])
metadatatype = numpy.dtype([ ('time_nr', numpy.int32), \
                             ('baseline', baselinetype), \
                             ('coordinate', coordinatetype)])
atermtype = numpy.complex64
spheroidaltype = numpy.float32

FourierDomainToImageDomain = 0
ImageDomainToFourierDomain = 1

import CPU
