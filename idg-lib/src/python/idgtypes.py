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
                             ('aterm_index', numpy.intc),
                             ('baseline', baselinetype),
                             ('coordinate', coordinatetype)])
atermtype = numpy.complex64
atermoffsettype = numpy.intc
spheroidaltype = numpy.float32

FourierDomainToImageDomain = 0
ImageDomainToFourierDomain = 1
