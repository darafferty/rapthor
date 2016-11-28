#include <iostream>
#include <cstdlib> // size_t
#include <complex>

#include "idg-cpu.h"
#include "idg-utility.h"  // Data init routines

using namespace std;

template <typename ProxyType>
void run()
{
    // Set constants explicitly in the parameters parameter
    clog << ">>> Configuration"  << endl;
    idg::Parameters params;
    // Read the following from ENV:
    // NR_STATIONS, NR_CHANNELS, NR_TIMESTEPS, NR_TIMESLOTS, IMAGESIZE,
    // GRIDSIZE
    // if non-default jobsize wanted, set also JOBSIZE, etc.
    params.set_from_env();

    // retrieve constants for memory allocation
    int nr_stations      = params.get_nr_stations();
    int nr_baselines     = params.get_nr_baselines();
    int nr_time          = params.get_nr_time();
    int nr_timeslots     = params.get_nr_timeslots();
    int nr_channels      = params.get_nr_channels();
    int gridsize         = params.get_grid_size();
    int subgridsize      = params.get_subgrid_size();
    int kernel_size      = params.get_kernel_size();
    float imagesize      = params.get_imagesize();
    int nr_polarizations = params.get_nr_polarizations();

    float w_offset = 0;

    // Print configuration
    clog << params;
    clog << endl;

    // Allocate and initialize data structures
    clog << ">>> Initialize data structures" << endl;

    auto size_visibilities = 1ULL * nr_baselines*nr_time*
                                    nr_channels*nr_polarizations;
    auto size_uvw          = 1ULL * nr_baselines*nr_time*3;
    auto size_wavenumbers  = 1ULL * nr_channels;
    auto size_aterm        = 1ULL * nr_stations*nr_timeslots*
                                    nr_polarizations*subgridsize*subgridsize;
    auto size_spheroidal   = 1ULL * subgridsize*subgridsize;
    auto size_grid         = 1ULL * nr_polarizations*gridsize*gridsize;
    auto size_baselines    = 1ULL * nr_baselines*2;

    auto visibilities  = (complex<float> *) malloc(sizeof(complex<float>) * size_visibilities);
    auto uvw           = (float *) malloc(sizeof(float) * size_uvw);
    auto wavenumbers   = (float *) malloc(sizeof(float) * size_wavenumbers);
    auto aterm         = (complex<float> *) malloc(sizeof(complex<float>) * size_aterm);
    auto aterm_offsets = (int *) malloc(sizeof(int) * nr_timeslots+1);
    auto spheroidal    = (float *) malloc(sizeof(float) * size_spheroidal);
    auto grid          = (complex<float> *) malloc(sizeof(complex<float>) * size_grid);
    auto baselines     = (int *) malloc(sizeof(int) * size_baselines);

    idg::init_example_visibilities(visibilities, nr_baselines,
                           nr_time,
                           nr_channels, nr_polarizations);
    idg::init_example_uvw(uvw, nr_stations, nr_baselines, nr_time);
    idg::init_example_wavenumbers(wavenumbers, nr_channels);
    idg::init_example_aterm(aterm, nr_timeslots, nr_stations, subgridsize, nr_polarizations);
    idg::init_example_aterm_offsets(aterm_offsets, nr_timeslots, nr_time);
    idg::init_example_spheroidal(spheroidal, subgridsize);
    idg::init_example_grid(grid, gridsize, nr_polarizations);
    idg::init_example_baselines(baselines, nr_stations, nr_baselines);
    clog << endl;

    // Initialize proxy
    clog << ">>> Initialize proxy" << endl;
    ProxyType proxy(params);
    clog << endl;

    // Run
    clog << ">>> Run gridding" << endl;
    proxy.grid_visibilities(visibilities, uvw, wavenumbers, baselines,
                            grid, w_offset, kernel_size, aterm, aterm_offsets,
                            spheroidal);

    clog << ">>> Run fft" << endl;
    proxy.transform(idg::FourierDomainToImageDomain, grid);

    clog << ">>> Run degridding" << endl;
    proxy.degrid_visibilities(visibilities, uvw, wavenumbers, baselines,
                              grid, w_offset, kernel_size, aterm, aterm_offsets,
                              spheroidal);

    // Free memory for data structures
    free(visibilities);
    free(uvw);
    free(wavenumbers);
    free(aterm);
    free(aterm_offsets);
    free(spheroidal);
    free(grid);
    free(baselines);
}
