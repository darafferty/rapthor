
#include <iostream>
#include <cstdlib> // size_t
#include <complex>

#include "idg-utility.h"  // Data init routines
#include "idg-opencl.h"

using namespace std;

int main(int argc, char **argv) {
    // Set constants explicitly in the parameters parameter
    clog << ">>> Configuration"  << endl;
    idg::Parameters params;
    params.set_from_env();

    // Retrieve constants for memory allocation
    int nr_stations = params.get_nr_stations();
    int nr_baselines = params.get_nr_baselines();
    int nr_time = params.get_nr_time();
    int nr_timeslots = params.get_nr_timeslots();
    int nr_channels = params.get_nr_channels();
    int gridsize = params.get_grid_size();
    int subgridsize = params.get_subgrid_size();
    float imagesize = params.get_imagesize();
    int nr_polarizations = params.get_nr_polarizations();

    float w_offset = 0;
    int kernel_size = (subgridsize / 4) + 1;

    // Print configuration
    clog << params;
    clog << endl;

    // Allocate and initialize data structures
    clog << ">>> Initialize data structures" << endl;

    auto size_visibilities = 1ULL * nr_baselines*nr_time*
        nr_channels*nr_polarizations;
    auto size_uvw = 1ULL * nr_baselines*nr_time*3;
    auto size_wavenumbers = 1ULL * nr_channels;
    auto size_aterm = 1ULL * nr_stations*nr_timeslots*
        nr_polarizations*subgridsize*subgridsize;
    auto size_spheroidal = 1ULL * subgridsize*subgridsize;
    auto size_grid = 1ULL * nr_polarizations*gridsize*gridsize;
    auto size_baselines = 1ULL * nr_baselines*2;

    auto visibilities = new complex<float>[size_visibilities];
    auto uvw = new float[size_uvw];
    auto wavenumbers = new float[size_wavenumbers];
    auto aterm = new complex<float>[size_aterm];
    auto aterm_offsets = new int[nr_timeslots+1];
    auto spheroidal = new float[size_spheroidal];
    auto grid = new complex<float>[size_grid];
    auto baselines = new int[size_baselines];

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
    idg::proxy::opencl::Generic proxy(params);
    clog << endl;

    // Run
    clog << ">>> Run fft" << endl;
    proxy.transform(idg::FourierDomainToImageDomain, grid);

    clog << ">>> Run gridding" << endl;
    proxy.grid_visibilities(visibilities, uvw, wavenumbers, baselines, grid, w_offset, kernel_size, aterm, aterm_offsets, spheroidal);

    clog << ">>> Run fft" << endl;
    proxy.transform(idg::ImageDomainToFourierDomain, grid);

    clog << ">>> Run degridding" << endl;
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
