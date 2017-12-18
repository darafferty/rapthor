#include <iostream>
#include <iomanip>
#include <cstdlib> // size_t
#include <complex>
#include <limits>

using namespace std;

#include "idg-cpu.h" // Reference proxy
#include "idg-util.h"  // Data init routines


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

void print_parameters(
    unsigned int nr_stations,
    unsigned int nr_channels,
    unsigned int nr_timesteps,
    unsigned int nr_timeslots,
    float image_size,
    unsigned int grid_size,
    unsigned int subgrid_size,
    unsigned int kernel_size
) {
    const int fw1 = 30;
    const int fw2 = 10;
    ostream &os = clog;

    os << "-----------" << endl;
    os << "PARAMETERS:" << endl;

    os << setw(fw1) << left << "Number of stations" << "== "
       << setw(fw2) << right << nr_stations << endl;

    os << setw(fw1) << left << "Number of channels" << "== "
       << setw(fw2) << right << nr_channels << endl;

    os << setw(fw1) << left << "Number of timesteps" << "== "
       << setw(fw2) << right << nr_timesteps << endl;

    os << setw(fw1) << left << "Number of timeslots" << "== "
       << setw(fw2) << right << nr_timeslots << endl;

    os << setw(fw1) << left << "Imagesize" << "== "
       << setw(fw2) << right << image_size  << endl;

    os << setw(fw1) << left << "Grid size" << "== "
       << setw(fw2) << right << grid_size << endl;

    os << setw(fw1) << left << "Subgrid size" << "== "
       << setw(fw2) << right << subgrid_size << endl;

    os << setw(fw1) << left << "Kernel size" << "== "
       << setw(fw2) << right << kernel_size << endl;

    os << "-----------" << endl;
}

// run gridding and degridding for ProxyType and reference CPU
// proxy and compare the outcome; usage run_test<proxy::cpu::Optimized>();
template <typename ProxyType>
int compare_to_reference(float tol = 1000*std::numeric_limits<float>::epsilon())
{
    int info = 0;

    // Parameters
    unsigned int nr_correlations = 4;
    float w_offset               = 0;
    unsigned int nr_stations     = 12;
    unsigned int nr_channels     = 9;
    unsigned int nr_timesteps    = 2048;
    unsigned int nr_timeslots    = 7;
    float image_size             = 0.08;
    unsigned int grid_size       = 1024;
    unsigned int subgrid_size    = 24;
    float cell_size              = image_size / grid_size;
    unsigned int kernel_size     = (subgrid_size / 2) + 1;
    unsigned int nr_baselines    = (nr_stations * (nr_stations - 1)) / 2;

    // Print parameters
    print_parameters(
        nr_stations, nr_channels, nr_timesteps, nr_timeslots,
        image_size, grid_size, subgrid_size, kernel_size);

    // Initialize proxies
    std::clog << ">>> Initialize proxy" << std::endl;
    ProxyType optimized;
    idg::proxy::cpu::Reference reference;
    std::clog << std::endl;

    // Allocate and initialize data structures
    clog << ">>> Initialize data structures" << endl;
    idg::Array1D<float> frequencies =
        idg::get_example_frequencies(nr_channels);
    idg::Array3D<idg::Visibility<std::complex<float>>> visibilities =
        idg::get_example_visibilities(nr_baselines, nr_timesteps, nr_channels);
    idg::Array3D<idg::Visibility<std::complex<float>>> visibilities_ref =
        idg::get_example_visibilities(nr_baselines, nr_timesteps, nr_channels);
    idg::Array1D<std::pair<unsigned int,unsigned int>> baselines =
        idg::get_example_baselines(nr_stations, nr_baselines);
    idg::Array2D<idg::UVWCoordinate<float>> uvw =
        idg::get_example_uvw(nr_stations, nr_baselines, nr_timesteps);
    idg::Array3D<std::complex<float>> grid =
        idg::get_zero_grid(nr_correlations, grid_size, grid_size);
    idg::Array3D<std::complex<float>> grid_ref =
        idg::get_zero_grid(nr_correlations, grid_size, grid_size);
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms =
        idg::get_identity_aterms(nr_timeslots, nr_stations, subgrid_size, subgrid_size);
    idg::Array1D<unsigned int> aterms_offsets =
        idg::get_example_aterms_offsets(nr_timeslots, nr_timesteps);
    idg::Array2D<float> spheroidal =
        idg::get_example_spheroidal(subgrid_size, subgrid_size);
    clog << endl;

    // Create plan
    clog << ">>> Create plan" << endl;
    idg::Plan plan(
        kernel_size, subgrid_size, grid_size, cell_size,
        frequencies, uvw, baselines, aterms_offsets);
    clog << endl;

    // Run gridder
    std::clog << ">>> Run gridding" << std::endl;
    optimized.gridding(
         plan, w_offset, cell_size, kernel_size, subgrid_size,
         frequencies, visibilities, uvw, baselines,
         grid, aterms, aterms_offsets, spheroidal);

    std::clog << ">>> Run reference gridding" << std::endl;
    reference.gridding(
         plan, w_offset, cell_size, kernel_size, subgrid_size,
         frequencies, visibilities, uvw, baselines,
         grid_ref, aterms, aterms_offsets, spheroidal);

    float grid_error = get_accucary(
        nr_correlations*grid_size*grid_size,
        grid.data(),
        grid_ref.data());


    // Run degridder
    std::clog << ">>> Run degridding" << std::endl;
    memset(visibilities.data(), 0, visibilities.bytes());
    memset(visibilities_ref.data(), 0, visibilities_ref.bytes());
    optimized.degridding(
        plan, w_offset, cell_size, kernel_size, subgrid_size,
        frequencies, visibilities, uvw, baselines,
        grid, aterms, aterms_offsets, spheroidal);

    std::clog << ">>> Run reference degridding" << std::endl;
    reference.degridding(
        plan, w_offset, cell_size, kernel_size, subgrid_size,
        frequencies, visibilities_ref, uvw, baselines,
        grid, aterms, aterms_offsets, spheroidal);

    std::clog << std::endl;

    // Ignore visibilities that are not included in the plan
    plan.mask_visibilities(visibilities);
    plan.mask_visibilities(visibilities_ref);

    float degrid_error = get_accucary(
        nr_baselines*nr_timesteps*nr_channels*nr_correlations,
        (std::complex<float> *) visibilities.data(),
        (std::complex<float> *) visibilities_ref.data());


    // Report results
    tol = grid_size * std::numeric_limits<float>::epsilon();
    if (grid_error < tol) {
        std::cout << "Gridding test PASSED!" << std::endl;
    } else {
        std::cout << "Gridding test FAILED!" << std::endl;
        info = 1;
    }

    tol = nr_baselines * nr_timesteps * nr_channels * std::numeric_limits<float>::epsilon();
    if (degrid_error < tol) {
        std::cout << "Degridding test PASSED!" << std::endl;
    } else {
        std::cout << "Degridding test FAILED!" << std::endl;
        info = 2;
    }

    std::cout << "grid_error = " << std::scientific << grid_error << std::endl;
    std::cout << "degrid_error = " << std::scientific << degrid_error << std::endl;

    return info;
}
