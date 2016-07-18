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
    visibilities =  numpy.zeros(
        (nr_baselines, nr_time, nr_channels, nr_polarizations),
        dtype = idg.visibilitiestype)

    # uvw
    uvw = idg.utils.get_example_uvw(nr_baselines, nr_time, integration_time)
    idg.utils.plot_uvw(uvw)

    # wavenumbers
    wavenumbers = idg.utils.get_example_wavenumbers(nr_channels)
    #idg.utils.plot_wavenumbers(wavenumbers)

    # baselines
    baselines = idg.utils.get_example_baselines(nr_baselines)

    # aterms
    aterms = idg.utils.get_example_aterms(nr_timeslots, nr_stations,
                                                 subgrid_size,
                                                 nr_polarizations)

    # aterm offset
    aterms_offset = idg.utils.get_example_aterms_offset(nr_timeslots,
                                                        nr_time)

    # spheroidal
    spheroidal = idg.utils.init_example_spheroidal_subgrid(subgrid_size)

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
