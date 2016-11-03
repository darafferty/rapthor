#include <iostream>
#include <iomanip>
#include <cstdlib> // size_t
#include <complex>
#include <limits>

#include "idg-cpu.h" // Reference proxy
#include "idg-utility.h"  // Data init routines


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
// proxy and compare the outcome; usage run_test<proxy::cpu::Optimized>();
template <typename ProxyType>
int compare_to_reference(float tol = 1000*std::numeric_limits<float>::epsilon())
{
    int info = 0;

    // Set constants explicitly in the parameters parameter
    std::clog << ">>> Configuration"  << std::endl;
    idg::Parameters params;
    params.set_nr_stations(12);
    params.set_nr_channels(9);
    params.set_nr_time(2048);
    params.set_nr_timeslots(7);
    params.set_imagesize(0.08);
    params.set_grid_size(1024);

    // retrieve constants for memory allocation
    int nr_stations      = params.get_nr_stations();
    int nr_baselines     = params.get_nr_baselines();
    int nr_time          = params.get_nr_time();
    int nr_timeslots     = params.get_nr_timeslots();
    int nr_channels      = params.get_nr_channels();
    int gridsize         = params.get_grid_size();
    int subgridsize      = params.get_subgrid_size();
    float imagesize      = params.get_imagesize();
    int nr_polarizations = params.get_nr_polarizations();
    float w_offset         = 0;
    float integration_time = 1.0f;
    int kernel_size        = (subgridsize / 2) + 1;

    // error tolerance, which might need to be adjusted if parameters are changed
    tol = gridsize * std::numeric_limits<float>::epsilon();

    // Print configuration
    std::clog << params;
    std::clog << std::endl;

    // Allocate and initialize data structures
    std::clog << ">>> Initialize data structures" << std::endl;

    auto size_visibilities = 1ULL * nr_baselines*nr_time*nr_channels*nr_polarizations;
    auto size_uvw          = 1ULL * nr_baselines*nr_time*3;
    auto size_wavenumbers  = 1ULL * nr_channels;
    auto size_aterm        = 1ULL * nr_timeslots*nr_stations*subgridsize*subgridsize*
                                    nr_polarizations;
    auto size_spheroidal   = 1ULL * subgridsize*subgridsize;
    auto size_grid         = 1ULL * nr_polarizations*gridsize*gridsize;
    auto size_baselines    = 1ULL * nr_baselines*2;

    auto visibilities     = new std::complex<float>[size_visibilities];
    auto visibilities_ref = new std::complex<float>[size_visibilities];
    auto uvw              = new float[size_uvw];
    auto wavenumbers      = new float[size_wavenumbers];
    auto aterm            = new std::complex<float>[size_aterm];
    auto aterm_offsets    = new int[nr_timeslots+1];
    auto spheroidal       = new float[size_spheroidal];
    auto grid             = new std::complex<float>[size_grid];
    auto grid_ref         = new std::complex<float>[size_grid];
    auto baselines        = new int[size_baselines];

    idg::init_example_visibilities(visibilities, nr_baselines, nr_time,
                                   nr_channels, nr_polarizations);
    idg::init_example_uvw(uvw, nr_stations, nr_baselines, nr_time, integration_time);
    idg::init_example_wavenumbers(wavenumbers, nr_channels);
    idg::init_example_aterm(aterm, nr_timeslots, nr_stations,
                            subgridsize, nr_polarizations);
    idg::init_example_aterm_offsets(aterm_offsets, nr_timeslots, nr_time);
    idg::init_example_spheroidal(spheroidal, subgridsize);
    idg::init_example_grid(grid, gridsize, nr_polarizations);
    idg::init_example_grid(grid_ref, gridsize, nr_polarizations);
    idg::init_example_baselines(baselines, nr_stations, nr_baselines);
    std::clog << std::endl;

    // Initialize interface to kernels
    std::clog << ">>> Initialize proxy" << std::endl;
    ProxyType optimized(params);
    idg::proxy::cpu::Reference reference(params);
    std::clog << std::endl;


    // Run gridder
    std::clog << ">>> Run gridding" << std::endl;
    optimized.grid_visibilities(
        visibilities, uvw, wavenumbers,
        baselines, grid, w_offset, kernel_size,
        aterm, aterm_offsets, spheroidal);

    std::clog << ">>> Run reference gridding" << std::endl;
    reference.grid_visibilities(
        visibilities, uvw, wavenumbers,
        baselines, grid_ref, w_offset, kernel_size,
        aterm, aterm_offsets, spheroidal);

    float grid_error = get_accucary(size_grid, grid, grid_ref);


    // Run degridder
    std::clog << ">>> Run degridding" << std::endl;
    optimized.degrid_visibilities(
        visibilities, uvw, wavenumbers,
        baselines, grid_ref, w_offset, kernel_size,
        aterm, aterm_offsets, spheroidal);

    std::clog << ">>> Run reference degridding" << std::endl;
    reference.degrid_visibilities(
        visibilities_ref, uvw, wavenumbers,
        baselines, grid_ref, w_offset, kernel_size,
        aterm, aterm_offsets, spheroidal);
    std::clog << std::endl;

    float degrid_error = get_accucary(size_visibilities,
                                      visibilities,
                                      visibilities_ref);


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
    delete[] visibilities_ref;
    delete[] uvw;
    delete[] wavenumbers;
    delete[] aterm;
    delete[] aterm_offsets;
    delete[] spheroidal;
    delete[] grid;
    delete[] grid_ref;

    return info;
}
