#include <iostream>
#include <iomanip>
#include <cstdlib> // size_t
#include <complex>
#include <tuple>
#include <typeinfo>

#include "idg-cpu.h"
#include "idg-utility.h"  // Data init routines

using namespace std;

std::tuple<int, int, int, int, float, int, int, int>read_parameters() {
    const unsigned int DEFAULT_NR_STATIONS = 44;
    const unsigned int DEFAULT_NR_CHANNELS = 8;
    const unsigned int DEFAULT_NR_TIME = 4096;
    const unsigned int DEFAULT_NR_TIMESLOTS = 16;
    const float DEFAULT_IMAGESIZE = 0.1f;
    const unsigned int DEFAULT_GRIDSIZE = 4096;
    const unsigned int DEFAULT_SUBGRIDSIZE = 24;

    char *cstr_nr_stations = getenv("NR_STATIONS");
    auto nr_stations = cstr_nr_stations ? atoi(cstr_nr_stations): DEFAULT_NR_STATIONS;

    char *cstr_nr_channels = getenv("NR_CHANNELS");
    auto nr_channels = cstr_nr_channels ? atoi(cstr_nr_channels) : DEFAULT_NR_CHANNELS;

    char *cstr_nr_time = getenv("NR_TIME");
    auto nr_time = cstr_nr_time ? atoi(cstr_nr_time) : DEFAULT_NR_TIME;

    char *cstr_nr_timeslots = getenv("NR_TIMESLOTS");
    auto nr_timeslots = cstr_nr_timeslots ? atoi(cstr_nr_timeslots) : DEFAULT_NR_TIMESLOTS;

    char *cstr_image_size = getenv("IMAGESIZE");
    auto image_size = cstr_image_size ? atof(cstr_image_size) : DEFAULT_IMAGESIZE;

    char *cstr_grid_size = getenv("GRIDSIZE");
    auto grid_size = cstr_grid_size ? atoi(cstr_grid_size) : DEFAULT_GRIDSIZE;

    char *cstr_subgrid_size = getenv("SUBGRIDSIZE");
    auto subgrid_size = cstr_subgrid_size ? atoi(cstr_subgrid_size) : DEFAULT_SUBGRIDSIZE;

    char *cstr_kernel_size = getenv("KERNELSIZE");
    auto kernel_size = cstr_kernel_size ? atoi(cstr_kernel_size) : (subgrid_size / 4) + 1;

    return std::make_tuple(
        nr_stations, nr_channels, nr_time, nr_timeslots,
        image_size, grid_size, subgrid_size, kernel_size);
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

template <typename ProxyType>
void run()
{
    // Constants
    unsigned int nr_correlations = 4;
    float w_offset = 0;
    unsigned int nr_stations;
    unsigned int nr_channels;
    unsigned int nr_timesteps;
    unsigned int nr_timeslots;
    float image_size;
    unsigned int grid_size;
    unsigned int subgrid_size;
    unsigned int kernel_size;
    
    // Read parameters from environment
    std::tie(
        nr_stations, nr_channels, nr_timesteps, nr_timeslots,
        image_size, grid_size, subgrid_size, kernel_size) = read_parameters();
    
    // Compute nr_baselines
    unsigned int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;

    // Compute cell_size
    float cell_size = image_size / grid_size;
    
    // Print parameters
    print_parameters(
        nr_stations, nr_channels, nr_timesteps, nr_timeslots,
        image_size, grid_size, subgrid_size, kernel_size);

    // Allocate and initialize data structures
    clog << ">>> Initialize data structures" << endl;
    idg::Array1D<float> frequencies =
        idg::get_example_frequencies(nr_channels);
    idg::Array3D<idg::Visibility<std::complex<float>>> visibilities =
        idg::get_example_visibilities(nr_baselines, nr_timesteps, nr_channels);
    idg::Array1D<std::pair<unsigned int,unsigned int>> baselines =
        idg::get_example_baselines(nr_stations, nr_baselines);
    idg::Array2D<idg::UVWCoordinate<float>> uvw =
        idg::get_example_uvw(nr_stations, nr_baselines, nr_timesteps);
    idg::Array3D<std::complex<float>> grid =
        idg::get_zero_grid(nr_correlations, grid_size, grid_size);
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms =
        idg::get_example_aterms(nr_timeslots, nr_stations, subgrid_size, subgrid_size);
    idg::Array1D<unsigned int> aterms_offsets =
        idg::get_example_aterms_offsets(nr_timeslots, nr_timesteps);
    idg::Array2D<float> spheroidal =
        idg::get_example_spheroidal(subgrid_size, subgrid_size);
    clog << endl;

    // Initialize proxy
    clog << ">>> Initialize proxy" << endl;
    idg::CompileConstants constants(nr_correlations, subgrid_size);
    ProxyType proxy(constants);
    clog << endl;

    // Run
    clog << ">>> Create plan" << endl;
    idg::Plan2 plan(
        kernel_size, subgrid_size, grid_size, cell_size,
        frequencies, uvw, baselines, aterms_offsets);
    clog << endl;

    clog << ">>> Run gridding" << endl;
    proxy.gridding(
        plan, w_offset, cell_size, kernel_size, frequencies, visibilities, uvw,
        baselines, grid, aterms, aterms_offsets, spheroidal);
    clog << endl;

    clog << ">>> Run fft" << endl;
    proxy.transform(idg::FourierDomainToImageDomain, grid);
    clog << endl;

    clog << ">>> Run degridding" << endl;
    proxy.degridding(
        plan, w_offset, cell_size, kernel_size, frequencies, visibilities, uvw,
        baselines, grid, aterms, aterms_offsets, spheroidal);
    clog << endl;
}
