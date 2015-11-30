#include <iostream>
#include <iomanip>
#include <cstdlib> // size_t
#include <complex>
#include <limits>

#include "Init.h"  // Data init routines


void run_gridding_test(
    const idg::Parameters& params,
    const int nr_subgrids,
    const float* uvw,
    const float* wavenumbers,
    const std::complex<float>* visibilities,
    const float* spheroidal,
    const std::complex<float>* aterm,
    const int* metadata,
    std::complex<float>* subgrids,
    std::complex<float>* subgrids_ref,
    std::complex<float>* grid,
    std::complex<float>* grid_ref);



void run_degridding_test(
    const idg::Parameters& params,
    const int nr_subgrids,
    const float* uvw,
    const float* wavenumbers,
    std::complex<float>* visibilities,
    std::complex<float>* visibilities_ref,
    const float* spheroidal,
    const std::complex<float>* aterm,
    const int* metadata,
    std::complex<float>* subgrids,
    const std::complex<float>* grid);


// compute max|A[i]-A_ref[i]| / max|A_ref[i]|
float get_accucary(
    const int size,
    const std::complex<float>* A,
    const std::complex<float>* A_ref)
{
    float max_abs_error = 0.0f;
    float max_ref_val = 0.0f;
    float max_val = 0.0f;
    for (int i=0; i<size; i++) {
        float abs_error = abs(A[i] - A_ref[i]);
        if ( abs_error > max_abs_error ) {
            max_abs_error = abs_error;
        }
        if (abs(A_ref[i]) > max_ref_val) {
            max_ref_val = abs(A_ref[i]);
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




int main(int argc, char *argv[])
{
    int info = 0;
    float tol = 100*std::numeric_limits<float>::epsilon();

    // Set constants explicitly in the parameters parameter
    std::clog << ">>> Configuration"  << std::endl;
    idg::Parameters params;
    // Read the following from ENV: 
    // NR_STATIONS, NR_CHANNELS, NR_TIMESTEPS, NR_TIMESLOTS, IMAGESIZE, 
    // GRIDSIZE 
    // if non-default jobsize wanted, set JOBSIZE (all routines), 
    // JOBSIZE_GRIDDING, JOBSIZE_DEGRIDDING, JOBSIZE_GRIDDER, 
    // JOBSIZE_ADDER, JOBSIZE_SPLITTER, JOBSIZE_DEGRIDDER
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

    // Compare gridding (onto subgrids and adding to grid) to reference
    run_gridding_test(params, nr_subgrids, uvw, wavenumbers, visibilities,  
                      spheroidal, aterm, metadata, subgrids, subgrids_ref, 
                      grid, grid_ref);

    // TODO: use data types, so that grid knows its size and we can write 
    // stuff like "grid_ref -= grid"; "grid_ref.maxabs()", ...
    float grid_error = get_accucary(size_grid, grid, grid_ref);
    if (grid_error < tol) {
        std::cout << "Gridding test PASSED!" << std::endl;
    } else {
        std::cout << "Gridding test FAILED!" << std::endl;
        info = 1;
    }


    // Compare degridding to reference
    run_degridding_test(params, nr_subgrids, uvw, wavenumbers, visibilities,
                        visibilities_ref, spheroidal, aterm, metadata,
                        subgrids, grid);

    // TODO: use data types, so that grid knows its size and we can write
    // stuff like "grid_ref -= grid"; "grid_ref.maxabs()", ...
    float degrid_error = get_accucary(size_grid, visibilities, visibilities_ref);
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
