#include <iostream>
#include <cstdlib> // size_t
#include <complex>

#include "idg-utility.h"  // Data init routines

template <typename PROXYNAME>
void run() {
    // Set constants explicitly in the parameters parameter
    std::clog << ">>> Configuration"  << std::endl;
    idg::Parameters params;
    // Read the following from ENV:
    // NR_STATIONS, NR_CHANNELS, NR_TIMESTEPS, NR_TIMESLOTS, IMAGESIZE,
    // GRIDSIZE
    // if non-default jobsize wanted, set also JOBSIZE, etc.
    params.set_from_env();

    // retrieve constants for memory allocation
    int nr_stations = params.get_nr_stations();
    int nr_baselines = params.get_nr_baselines();
    int nr_time =  params.get_nr_time();
    int nr_timeslots = params.get_nr_timeslots();
    int nr_channels = params.get_nr_channels();
    int gridsize = params.get_grid_size();
    int subgridsize = params.get_subgrid_size();
    float imagesize = params.get_imagesize();
    int nr_polarizations = params.get_nr_polarizations();

    float w_offset = 0;
    int kernel_size = (subgridsize / 4) + 1;

    // Print configuration
    std::clog << params;
    std::clog << std::endl;

    // Allocate and initialize data structures
    std::clog << ">>> Initialize data structures" << std::endl;

    auto size_visibilities = 1ULL * nr_baselines*nr_time*
        nr_channels*nr_polarizations;
    auto size_uvw = 1ULL * nr_baselines*nr_time*3;
    auto size_wavenumbers = 1ULL * nr_channels;
    auto size_aterm = 1ULL * nr_stations*nr_timeslots*
        nr_polarizations*subgridsize*subgridsize;
    auto size_spheroidal = 1ULL * subgridsize*subgridsize;
    auto size_grid = 1ULL * nr_polarizations*gridsize*gridsize;
    auto size_baselines = 1ULL * nr_baselines*2;

    auto visibilities = new std::complex<float>[size_visibilities];
    auto uvw = new float[size_uvw];
    auto wavenumbers = new float[size_wavenumbers];
    auto aterm = new std::complex<float>[size_aterm];
    auto aterm_offsets = new int[nr_timeslots+1];
    auto spheroidal = new float[size_spheroidal];
    auto grid = new std::complex<float>[size_grid];
    auto baselines = new int[size_baselines];

    idg::init_visibilities(visibilities, nr_baselines,
                           nr_time,
                           nr_channels, nr_polarizations);
    idg::init_uvw(uvw, nr_stations, nr_baselines, nr_time);
    idg::init_wavenumbers(wavenumbers, nr_channels);
    idg::init_aterm(aterm, nr_stations, nr_timeslots, nr_polarizations,
                    subgridsize);
    idg::init_aterm_offsets(aterm_offsets, nr_timeslots, nr_time);
    idg::init_spheroidal(spheroidal, subgridsize);
    idg::init_grid(grid, gridsize, nr_polarizations);
    idg::init_baselines(baselines, nr_stations, nr_baselines);
    std::clog << std::endl;

    // Initialize proxy
    std::clog << ">>> Initialize proxy" << std::endl;
    PROXYNAME proxy(params);
    std::clog << std::endl;

    // Run
    std::clog << ">>> Run gridding" << std::endl;
    proxy.grid_visibilities(visibilities, uvw, wavenumbers, baselines, grid, w_offset, kernel_size, aterm, aterm_offsets, spheroidal);

    std::clog << ">>> Run fft" << std::endl;
    proxy.transform(idg::FourierDomainToImageDomain, grid);

    std::clog << ">>> Run degridding" << std::endl;
    proxy.degrid_visibilities(visibilities, uvw, wavenumbers, baselines, grid, w_offset, kernel_size, aterm, aterm_offsets, spheroidal);

    // Free memory for data structures
    delete[] visibilities;
    delete[] uvw;
    delete[] wavenumbers;
    delete[] aterm;
    delete[] aterm_offsets;
    delete[] spheroidal;
    delete[] grid;
    delete[] baselines;
}
