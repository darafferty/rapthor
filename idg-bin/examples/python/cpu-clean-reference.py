#!/usr/bin/env python

# after INSTALLING the library, and sourcing init-enviroment.sh
# (or setting the PYTHONPATH manually), you can import the idg module
import idg
import numpy
import matplotlib.pyplot as plt
import sys
import signal


def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


def create_images(p, visibilities, uvw, wavenumbers,
                  baselines, w_offset, kernel_size,
                  aterms, aterms_offset, spheroidal):
    ##########
    # Allocate
    ##########
    dirty = numpy.zeros((nr_polarizations, grid_size, grid_size),
                        dtype = idg.gridtype)
    residual = numpy.zeros((nr_polarizations, grid_size, grid_size),
                           dtype = idg.gridtype)
    clean = numpy.zeros((nr_polarizations, grid_size, grid_size),
                        dtype = idg.gridtype)

    ##########
    # Find PSF
    ##########
    psf_visibilities =  numpy.ones(
        (nr_baselines, nr_time, nr_channels, nr_polarizations),
        dtype = idg.visibilitiestype)
    psf = numpy.zeros((nr_polarizations, grid_size, grid_size),
                      dtype = idg.gridtype)
    p.grid_visibilities(psf_visibilities, uvw, wavenumbers, baselines, psf,
                        w_offset, kernel_size, aterms, aterms_offset,
                        spheroidal)
    p.transform(idg.FourierDomainToImageDomain, psf)
    # idg.utils.plot_grid(psf)

    gain = 0.2   # gain, recommended 0.1 - 0.2
    max_iter = 3;
    residual_visibilities =  numpy.zeros(
        (nr_baselines, nr_time, nr_channels, nr_polarizations),
        dtype = idg.visibilitiestype)
    for i in range(max_iter):

        ###############
        # Find residual
        ###############
        # idg.utils.plot_visibilities(visibilities - residual_visibilities)

        p.grid_visibilities(visibilities - residual_visibilities, uvw,
                            wavenumbers, baselines, residual,
                            w_offset, kernel_size, aterms, aterms_offset,
                            spheroidal)
        p.transform(idg.FourierDomainToImageDomain, residual)

        # idg.utils.plot_grid(residual)

        ##################
        # Find dirty image
        ##################
        if i==0:
            dirty = numpy.copy(residual)

        ####################
        # Find maximal value
        ####################
        (pol_pol,i_max,j_max) = numpy.unravel_index(residual.argmax(), residual.shape)
        # I_max = numpy.real(dirty[pol_pol,i_max,j_max])

        ######################
        # Put into clean image
        ######################
        clean[:,i_max,j_max] = clean[:,i_max,j_max] + gain*residual[:,i_max,j_max]

        # idg.utils.plot_grid(clean)

        ####################
        # Degrid model image
        ####################
        p.transform(idg.ImageDomainToFourierDomain, clean)

        p.degrid_visibilities(residual_visibilities, uvw, wavenumbers,
                              baselines, clean, w_offset,
                              kernel_size, aterms, aterms_offset,
                              spheroidal)

        scale = numpy.mean(residual_visibilities)
        residual_visibilities = residual_visibilities*  (gain/scale)
        idg.utils.plot_visibilities(residual_visibilities)

    plt.show()

    return clean, dirty, residual




if __name__ == "__main__":
    ############
    # paramaters
    ############
    nr_stations = 8
    nr_baselines = nr_stations*(nr_stations-1)/2
    nr_channels = 1
    nr_time = 4            # samples per baseline
    nr_timeslots = 1       # A-term time slots
    image_size = 0.08
    subgrid_size = 24
    grid_size = 512
    integration_time = 10
    kernel_size = (subgrid_size / 2) + 1

    ##################
    # initialize proxy
    ##################
    p = idg.CPU.Reference(nr_stations, nr_channels,
                          nr_time, nr_timeslots,
                          image_size, grid_size, subgrid_size)

    ##################
    # print parameters
    ##################
    print "nr_stations = ", p.get_nr_stations()
    print "nr_baselines = ", p.get_nr_baselines()
    print "nr_channels = ", p.get_nr_channels()
    print "nr_timeslots = ", p.get_nr_timeslots()
    print "nr_polarizations = ", p.get_nr_polarizations()
    print "subgrid_size = ", p.get_subgrid_size()
    print "grid_size = ", p.get_grid_size()
    print "image_size = ", p.get_image_size()
    print "kernel_size = ", kernel_size
    print "job size for gridding = ", p.get_job_size_gridding()
    print "job size for degridding = ", p.get_job_size_degridding()

    #################
    # initialize data
    #################

    # allocate memory for data
    nr_polarizations = p.get_nr_polarizations()

    # visibilities
    visibilities =  numpy.ones(
        (nr_baselines, nr_time, nr_channels, nr_polarizations),
        dtype = idg.visibilitiestype)
    idg.utils.init_visibilities(visibilities)
    # idg.utils.plot_visibilities(visibilities)

    # uvw
    uvw = numpy.zeros((nr_baselines, nr_time),
                      dtype = idg.uvwtype)
    idg.utils.init_uvw(uvw, integration_time)
    # idg.utils.plot_uvw(uvw)

    # wavenumbers
    wavenumbers = numpy.ones(nr_channels,
                             dtype = idg.wavenumberstype)
    idg.utils.init_wavenumbers(wavenumbers)
    #idg.utils.plot_wavenumbers(wavenumbers)

    # baselines
    baselines = numpy.zeros(nr_baselines, dtype = idg.baselinetype)
    idg.utils.init_baselines(baselines)

    # aterms
    aterms = numpy.zeros((nr_stations, nr_timeslots, nr_polarizations,
                          subgrid_size, subgrid_size), \
                         dtype = idg.atermtype)
    # idg.utils.init_aterms(aterms)
    # TODO: update C++ init_aterms
    # Set aterm to identity instead
    aterms[:,:,0,:,:] = 1.0
    aterms[:,:,3,:,:] = 1.0

    # aterm offset
    aterms_offset = numpy.zeros((nr_timeslots + 1), dtype = idg.atermoffsettype)
    idg.utils.init_aterms_offset(aterms_offset, nr_time)

    # spheroidal
    spheroidal = numpy.ones((subgrid_size, subgrid_size),
                             dtype = idg.spheroidaltype)
    idg.utils.init_spheroidal(spheroidal)
    #idg.utils.plot_spheroidal(spheroidal)

    # metadata (for debugging)
    nr_subgrids = p._get_nr_subgrids(uvw, wavenumbers, baselines, aterms_offset, kernel_size)
    metadata = numpy.zeros(nr_subgrids, dtype = idg.metadatatype)
    p._init_metadata(metadata, uvw, wavenumbers, baselines, aterms_offset, kernel_size)
    idg.utils.plot_metadata(metadata, uvw, wavenumbers, grid_size, subgrid_size, image_size)

    print metadata[0]
    print uvw[0]
    print uvw[1]
    print uvw[12]
    print uvw[13]
    print uvw[25]
    print uvw[26]


    w_offset = 0.0

    create_images(p, visibilities, uvw, wavenumbers,
                  baselines, w_offset, kernel_size,
                  aterms, aterms_offset, spheroidal)
