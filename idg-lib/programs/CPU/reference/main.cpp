#include <iostream>
#include <cstdlib> // size_t
#include <complex>
#include "Proxy.h"
#include "Init.h"  // Data init routines
#include "Arguments.h"  // Parse command line arguments

using namespace std;

int main(int argc, char *argv[]) {
    // Set constants explicitly in the parameters parameter
    clog << ">>> Configuration"  << endl;
    idg::Parameters params;
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
    clog << params;
    clog << endl;

    // Allocate and initialize data structures
    clog << ">>> Initialize data structures" << endl;

    auto size_visibilities = 1ULL * nr_baselines*nr_timesteps*nr_timeslots*nr_channels*nr_polarizations;
    auto size_uvw = 1ULL * nr_baselines*nr_timesteps*nr_timeslots*3;
    auto size_wavenumbers = 1ULL * nr_channels;
    auto size_aterm = 1ULL * nr_stations*nr_timeslots*nr_polarizations*subgridsize*subgridsize;
    auto size_spheroidal = 1ULL * subgridsize*subgridsize;
    auto size_grid = 1ULL * nr_polarizations*gridsize*gridsize;
    auto size_metadata = 1ULL * nr_subgrids*5;
    auto size_subgrids = 1ULL * nr_subgrids*nr_polarizations*subgridsize*subgridsize;

    auto visibilities = new complex<float>[size_visibilities];
    auto uvw = new float[size_uvw];
    auto wavenumbers = new float[size_wavenumbers];
    auto aterm = new complex<float>[size_aterm];
    auto spheroidal = new float[size_spheroidal];
    auto grid = new complex<float>[size_grid];
    auto metadata = new int[size_metadata];
    auto subgrids = new complex<float>[size_subgrids];

    idg::init_visibilities(visibilities, nr_baselines, nr_timesteps*nr_timeslots, nr_channels, nr_polarizations);
    idg::init_uvw(uvw, nr_stations, nr_baselines, nr_timesteps*nr_timeslots, gridsize, subgridsize);
    idg::init_wavenumbers(wavenumbers, nr_channels);
    idg::init_aterm(aterm, nr_stations, nr_timesteps, nr_polarizations, subgridsize);
    idg::init_spheroidal(spheroidal, subgridsize);
    idg::init_grid(grid, gridsize, nr_polarizations);
    idg::init_metadata(metadata, uvw, wavenumbers, nr_stations, nr_baselines, nr_timesteps, nr_timeslots, nr_channels, gridsize, subgridsize, imagesize);

    clog << endl;


    // Initialize interface to kernels
    clog << ">>> Initialize proxy" << endl;

    // basic gcc settings
    idg::Compiler compiler = "g++";
    idg::Compilerflags compilerflags = "-Wall -O3 -fopenmp -lfftw3 -lfftw3f -lfftw3f_omp";

    idg::proxy::CPU xeon(compiler, compilerflags, params);
    clog << endl;

    // Run gridder
    clog << ">>> Run gridder" << endl;
    int jobsize_gridder = params.get_job_size_gridder();
    xeon.grid_onto_subgrids(jobsize_gridder, nr_subgrids, 0, uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrids);

    clog << ">> Run adder" << endl;
    int jobsize_adder = params.get_job_size_adder();
    xeon.add_subgrids_to_grid(jobsize_adder, nr_subgrids, metadata, subgrids, grid);

    clog << "Run fft" << endl;
    xeon.transform(idg::FourierDomainToImageDomain, grid);

    clog << ">>> Run splitter" << endl;
    int jobsize_splitter = params.get_job_size_splitter();
    xeon.split_grid_into_subgrids(jobsize_splitter, nr_subgrids, metadata, subgrids, grid);

    clog << ">>> Run degridder" << endl;
    int jobsize_degridder = params.get_job_size_degridder();
    xeon.degrid_from_subgrids(jobsize_degridder, nr_subgrids, 0, uvw, wavenumbers, visibilities, spheroidal, aterm, metadata, subgrids);

    // free memory for data structures
    delete[] visibilities;
    delete[] uvw;
    delete[] wavenumbers;
    delete[] aterm;
    delete[] spheroidal;
    delete[] grid;
    delete[] subgrids;
    delete[] metadata;

    return EXIT_SUCCESS;
}
