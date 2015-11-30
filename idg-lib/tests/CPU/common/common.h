#include <iostream>
#include <iomanip>
#include <cstdlib> // size_t
#include <complex>
#include <limits>

#include "CPU/Reference/idg.h" // Reference proxy
#include "Init.h"  // Data init routines


// computes max|A[i]-B[i]| / max|B[i]|
float get_accucary(
    const int size,
    const std::complex<float>* A,
    const std::complex<float>* B)
{
    float max_abs_error = 0.0f;
    float max_ref_val = 0.0f;
    float max_val = 0.0f;
    for (int i=0; i<size; i++) {
        float abs_error = abs(A[i] - B[i]);
        if ( abs_error > max_abs_error ) {
            max_abs_error = abs_error;
        }
        if (abs(B[i]) > max_ref_val) {
            max_ref_val = abs(B[i]);
        }
        if (abs(A[i]) > max_val) {
            max_val = abs(A[i]);
        }
    }
    if (max_ref_val == 0.0f) {
        if (max_val == 0.0f)
            // both grid are zero
            return 0.0f;
        else
            // refrence grid is zero, but computed grid not
            return std::numeric_limits<float>::infinity();
    } else {
        return max_abs_error / max_ref_val;
    }
}


// run gridding and degridding for ProxyType and reference CPU
// proxy and compare the outcome; usage run_test<proxy::cpu::HaswellEP>();
template <typename ProxyType>
int run_test()
{
    int info = 0;
    float tol = 100*std::numeric_limits<float>::epsilon();

    // Set constants explicitly in the parameters parameter
    std::clog << ">>> Configuration"  << std::endl;
    idg::Parameters params;
    // Read the following from ENV: 
    // NR_STATIONS, NR_CHANNELS, NR_TIMESTEPS, NR_TIMESLOTS, IMAGESIZE, 
    // GRIDSIZE; if non-default jobsize wanted, set jobsizes!
    params.set_from_env();

    // retrieve constants for memory allocation
    int nr_stations = params.get_nr_stations();
    int nr_baselines = params.get_nr_baselines();
    int nr_timesteps = params.get_nr_timesteps();
    int nr_timeslots = params.get_nr_timeslots();
    int nr_channels = params.get_nr_channels();
    int gridsize = params.get_grid_size();
    int subgridsize = params.get_subgrid_size();
    float imagesize = params.get_imagesize();
    int nr_polarizations = 4;
    int nr_subgrids = nr_baselines * nr_timeslots;

    // Print configuration
    std::clog << params;
    std::clog << std::endl;

    // Allocate and initialize data structures
    std::clog << ">>> Initialize data structures" << std::endl;

    auto size_visibilities = 1ULL * nr_baselines*nr_timesteps*nr_timeslots*nr_channels*nr_polarizations;
    auto size_uvw = 1ULL * nr_baselines*nr_timesteps*nr_timeslots*3;
    auto size_wavenumbers = 1ULL * nr_channels;
    auto size_aterm = 1ULL * nr_stations*nr_timeslots*nr_polarizations*subgridsize*subgridsize;
    auto size_spheroidal = 1ULL * subgridsize*subgridsize;
    auto size_grid = 1ULL * nr_polarizations*gridsize*gridsize;
    auto size_metadata = 1ULL * nr_subgrids*5;
    auto size_subgrids = 1ULL * nr_subgrids*nr_polarizations*subgridsize*subgridsize;

    auto visibilities = new std::complex<float>[size_visibilities];
    auto visibilities_ref = new std::complex<float>[size_visibilities];
    auto uvw = new float[size_uvw];
    auto wavenumbers = new float[size_wavenumbers];
    auto aterm = new std::complex<float>[size_aterm];
    auto spheroidal = new float[size_spheroidal];
    auto grid = new std::complex<float>[size_grid];
    auto grid_ref = new std::complex<float>[size_grid];
    auto metadata = new int[size_metadata];
    auto subgrids = new std::complex<float>[size_subgrids];
    auto subgrids_ref = new std::complex<float>[size_subgrids];

    idg::init_visibilities(visibilities, nr_baselines, nr_timesteps*nr_timeslots,
                           nr_channels, nr_polarizations);
    idg::init_visibilities(visibilities_ref, nr_baselines, nr_timesteps*nr_timeslots,
                           nr_channels, nr_polarizations);
    idg::init_uvw(uvw, nr_stations, nr_baselines, nr_timesteps*nr_timeslots, gridsize, subgridsize);
    idg::init_wavenumbers(wavenumbers, nr_channels);
    idg::init_aterm(aterm, nr_stations, nr_timeslots, nr_polarizations, subgridsize);
    idg::init_spheroidal(spheroidal, subgridsize);
    idg::init_grid(grid, gridsize, nr_polarizations);
    idg::init_grid(grid_ref, gridsize, nr_polarizations);
    idg::init_metadata(metadata, uvw, wavenumbers, nr_stations, nr_baselines,
                       nr_timesteps, nr_timeslots, nr_channels, gridsize, subgridsize, imagesize);
    std::clog << std::endl;

    // Initialize interface to kernels
    std::clog << ">>> Initialize proxy" << std::endl;
    ProxyType optimized(params);
    idg::proxy::cpu::Reference reference(params);
    std::clog << std::endl;


    // Run gridder
    std::clog << ">>> Run gridder" << std::endl;
    optimized.grid_onto_subgrids(nr_subgrids, 0, uvw, wavenumbers,
                                 visibilities, spheroidal, aterm,
                                 metadata, subgrids);

    std::clog << ">>> Run adder" << std::endl;
    optimized.add_subgrids_to_grid(nr_subgrids, metadata,
                                   subgrids, grid);

    std::clog << ">>> Run reference gridder" << std::endl;
    reference.grid_onto_subgrids(nr_subgrids, 0, uvw, wavenumbers,
                                 visibilities, spheroidal, aterm,
                                 metadata, subgrids_ref);

    std::clog << ">>> Run reference adder" << std::endl;
    reference.add_subgrids_to_grid(nr_subgrids, metadata,
                                   subgrids_ref, grid_ref);

    float grid_error = get_accucary(size_grid, grid, grid_ref);


    // Run degridder
    std::clog << ">>> Run splitter" << std::endl;
    // uses reference grid
    optimized.split_grid_into_subgrids(nr_subgrids, metadata,
                                       subgrids, grid_ref);

    std::clog << ">>> Run degridder" << std::endl;
    optimized.degrid_from_subgrids(nr_subgrids, 0, uvw, wavenumbers,
                                   visibilities, spheroidal, aterm,
                                   metadata, subgrids);

    std::clog << ">>> Run reference splitter" << std::endl;
    reference.split_grid_into_subgrids(nr_subgrids, metadata,
                                       subgrids_ref, grid_ref);

    std::clog << ">>> Run reference degridder" << std::endl;
    reference.degrid_from_subgrids(nr_subgrids, 0, uvw, wavenumbers,
                                   visibilities_ref, spheroidal, aterm,
                                   metadata, subgrids_ref);
    std::clog << std::endl;


    float degrid_error = get_accucary(size_grid, visibilities, visibilities_ref);


    // Report results

    if (grid_error < tol) {
        std::cout << "Gridding test PASSED!" << std::endl;
    } else {
        std::cout << "Gridding test FAILED!" << std::endl;
        info = 1;
    }

    if (degrid_error < tol) {
        std::cout << "Degridding test PASSED!" << std::endl;
    } else {
        std::cout << "Degridding test FAILED!" << std::endl;
        info = 2;
    }

    std::cout << "grid_error = " << std::scientific << grid_error << std::endl;
    std::cout << "degrid_error = " << std::scientific << degrid_error << std::endl;


    // Free memory for data structures
    delete[] visibilities;
    delete[] uvw;
    delete[] wavenumbers;
    delete[] aterm;
    delete[] spheroidal;
    delete[] grid;
    delete[] grid_ref;
    delete[] subgrids;
    delete[] subgrids_ref;
    delete[] metadata;

    return info;
}
