import os
import ctypes
import numpy
from ctypes.util import find_library

try:
    import CPU
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
baselinetype = numpy.dtype([('station1', numpy.int32),
                            ('station2', numpy.int32)])
coordinatetype = numpy.dtype([('x', numpy.int32),
                              ('y', numpy.int32)])
metadatatype = numpy.dtype([ ('time_nr', numpy.int32),
                             ('baseline', baselinetype),
                             ('coordinate', coordinatetype)])
atermtype = numpy.complex64
spheroidaltype = numpy.float32

FourierDomainToImageDomain = 0
ImageDomainToFourierDomain = 1


class Parameters(object):

    def __init__(self, **kwargs):
        print kwargs
        for a in kwargs:
            self.nr_stations = kwargs.get('nr_stations')
            self.nr_channels = kwargs.get('nr_channels')
            self.nr_timeslots = kwargs.get('nr_timeslots')
            self.nr_timesteps = kwargs.get('nr_timesteps')
            self.nr_polarizations = kwargs.get('nr_polarizations')
            self.image_size = kwargs.get('image_size')
            self.grid_size = kwargs.get('grid_size')
            self.subgrid_size = kwargs.get('subgrid_size')
            self.job_size = kwargs.get('job_size')
