#include <iostream>
#include <cstdlib> 
#include <complex>

#include <cuda.h>
#include <cudaProfiler.h>

#include "idg-utility.h"  // Data init routines

using namespace std;

void printDevices(int deviceNumber) {
	std::clog << "Devices";
	for (int device = 0; device < cu::Device::getCount(); device++) {
		std::clog << "\t" << device << ": ";
		std::clog << cu::Device(device).getName();
		if (device == deviceNumber) {
			std::clog << "\t" << "<---";
		}
		std::clog << std::endl;
	}
	std::clog << "\n";
}

void run(
    idg::Parameters params,
    unsigned deviceNumber,
    cu::Context &context,
    int nr_subgrids,
    cu::HostMemory &h_uvw,
    cu::DeviceMemory &d_wavenumbers,
    cu::HostMemory &h_visibilities,
    cu::DeviceMemory &d_spheroidal,
    cu::DeviceMemory &d_aterm,
    cu::HostMemory &h_metadata,
    cu::HostMemory &h_subgrids,
    cu::HostMemory &h_grid
);

template <typename PROXYNAME>
void run() {
    // Set constants explicitly in the parameters parameter
    clog << ">>> Configuration"  << endl;
    idg::Parameters params;
    params.set_from_env();

    // Get device number
    char *cstr_deviceNumber = getenv("CUDA_DEVICE");
    unsigned deviceNumber = cstr_deviceNumber ? atoi (cstr_deviceNumber) : 0;

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

    // Initialize interface to kernels
    clog << ">>> Initialize proxy" << endl;
    PROXYNAME proxy(params, deviceNumber);
    clog << endl;

    // Get context from proxy
    cu::Context &context = proxy.get_context();

    // Show CUDA devices
    printDevices(deviceNumber);

    // Allocate data structures
    clog << ">>> Allocate data structures" << endl;
    auto size_visibilities = 1ULL * nr_baselines*nr_timesteps*nr_timeslots*nr_channels*nr_polarizations;
    auto size_uvw = 1ULL * nr_baselines*nr_timesteps*nr_timeslots*3;
    auto size_wavenumbers = 1ULL * nr_channels;
    auto size_aterm = 1ULL * nr_stations*nr_timeslots*nr_polarizations*subgridsize*subgridsize;
    auto size_spheroidal = 1ULL * subgridsize*subgridsize;
    auto size_grid = 1ULL * nr_polarizations*gridsize*gridsize;
    auto size_metadata = 1ULL * nr_subgrids*5;
    auto size_subgrids = 1ULL * nr_subgrids*nr_polarizations*subgridsize*subgridsize;

    cu::HostMemory h_visibilities(sizeof(complex<float>) * size_visibilities);
    cu::HostMemory h_uvw(sizeof(float) * size_uvw);
    cu::DeviceMemory d_wavenumbers(sizeof(float) * size_wavenumbers);
    cu::DeviceMemory d_aterm(sizeof(complex<float>) * size_aterm);
    cu::DeviceMemory d_spheroidal(sizeof(float) * size_spheroidal);
    cu::HostMemory h_grid(sizeof(complex<float>) * size_grid);
    cu::HostMemory h_metadata(sizeof(int) * size_metadata);
    cu::HostMemory h_subgrids(sizeof(complex<float>) * size_subgrids);

    clog << ">>> Initialize data structures" << endl;
    void *wavenumbers = idg::init_wavenumbers(nr_channels);
    void *aterm       = idg::init_aterm(nr_stations, nr_timeslots, nr_polarizations, subgridsize);
    void *spheroidal  = idg::init_spheroidal(subgridsize);
    idg::init_visibilities(h_visibilities, nr_baselines, nr_timesteps*nr_timeslots, nr_channels, nr_polarizations);
    idg::init_uvw(h_uvw, nr_stations, nr_baselines, nr_timesteps*nr_timeslots);
    idg::init_metadata(h_metadata, h_uvw, wavenumbers, nr_stations, nr_baselines, nr_timesteps, nr_timeslots, nr_channels, gridsize, subgridsize, imagesize);
    idg::init_grid(h_grid, gridsize, nr_polarizations);
    d_wavenumbers.set(wavenumbers);
    d_aterm.set(aterm);
    d_spheroidal.set(spheroidal);

    // Start profiling
    cuProfilerStart();

    // Run gridder
    clog << ">>> Run gridder" << endl;
    proxy.grid_onto_subgrids(context, nr_subgrids, 0, h_uvw, d_wavenumbers, h_visibilities, d_spheroidal, d_aterm, h_metadata, h_subgrids);

    // Run adder
    clog << ">>> Run adder" << endl;
    proxy.add_subgrids_to_grid(context, nr_subgrids, h_metadata, h_subgrids, h_grid);

    // Run fft
    clog << ">>> Run fft" << endl;
    proxy.transform(idg::FourierDomainToImageDomain, context, h_grid);

    // Run splitter
    clog << ">>> Run splitter" << endl;
    proxy.split_grid_into_subgrids(context, nr_subgrids, h_metadata, h_subgrids, h_grid);

    // Run degridder
    clog << ">>> Run degridder" << endl;
    proxy.degrid_from_subgrids(context, nr_subgrids, 0, h_uvw, d_wavenumbers, h_visibilities, d_spheroidal, d_aterm, h_metadata, h_subgrids);

    // Stop profiling
    cuProfilerStop();

    // Cleanup
    free(wavenumbers);
    free(aterm);
    free(spheroidal);
}
