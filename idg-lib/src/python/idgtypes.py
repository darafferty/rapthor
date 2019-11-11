import numpy

visibilitiestype = numpy.complex64
uvwtype = numpy.dtype([('u', numpy.float32),
                       ('v', numpy.float32),
                       ('w', numpy.float32)])
frequenciestype = numpy.float32
gridtype = numpy.complex64
baselinetype = numpy.dtype([('station1', numpy.intc),
                            ('station2', numpy.intc)])
coordinatetype = numpy.dtype([('x', numpy.intc),
                              ('y', numpy.intc),
                              ('z', numpy.intc)])
metadatatype = numpy.dtype([ ('time_index', numpy.intc),
                             ('nr_timesteps', numpy.intc),
                             ('channel_begin', numpy.intc),
                             ('channel_end', numpy.intc),
                             ('baseline', baselinetype),
                             ('coordinate', coordinatetype),
                             ('wtile_coordinate', coordinatetype),
                             ('wtile_index', numpy.intc),
                             ('nr_aterms', numpy.intc)])
atermtype = numpy.complex64
atermoffsettype = numpy.intc
spheroidaltype = numpy.float32

FourierDomainToImageDomain = 0
ImageDomainToFourierDomain = 1
