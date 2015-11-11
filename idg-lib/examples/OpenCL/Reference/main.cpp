#include <iostream>
#include <cstdlib> // size_t
#include <complex>

#include <opencl.h>

#include "OpenCL/Reference/idg.h"

#include "Init.h"  // Data init routines

using namespace std;

void printDevices(cl::Context &context, int deviceNumber) {
	// Get devices
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	
	std::clog << "Devices" << std::endl;
	for (int d = 0; d < devices.size(); d++) {
		cl::Device device = devices[d];
		std::clog << "Device: "			  << device.getInfo<CL_DEVICE_NAME>();
		if (d == deviceNumber) {
			std::clog << "\t" << "<---";
		}
		std::clog << std::endl;
		std::clog << "Driver version  : " << device.getInfo<CL_DRIVER_VERSION>() << std::endl;
		std::clog << "Compute units   : " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
		std::clog << "Clock frequency : " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << " MHz" << std::endl;
        std::clog << "Global memory   : " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() * 1e-9 << " Gb" << std::endl;
        std::clog << std::endl;
    }
	std::clog << "\n";
}

int main(int argc, char *argv[]) {
    // Set constants explicitly in the parameters parameter
    clog << ">>> Configuration"  << endl;
    idg::Parameters params;
    params.set_from_env();

    // Get device number
    char *cstr_deviceNumber = getenv("OPENCL_DEVICE");
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

    // Initialize OpenCL
    std::clog << ">>> Initialize OpenCL" << std::endl;
    cl::Context context = cl::Context(CL_DEVICE_TYPE_ALL);
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::Device device = devices[deviceNumber];
    cl::CommandQueue queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Show OpenCL devices
    printDevices(context, deviceNumber);

    // Allocate data structures
    clog << ">>> Allocate data structures" << endl;
    auto size_visibilities = 1ULL * nr_baselines*nr_timesteps*nr_timeslots*nr_channels*nr_polarizations*sizeof(complex<float>);
    auto size_uvw = 1ULL * nr_baselines*nr_timesteps*nr_timeslots*3*sizeof(float);
    auto size_wavenumbers = 1ULL * nr_channels*sizeof(float);
    auto size_aterm = 1ULL * nr_stations*nr_timeslots*nr_polarizations*subgridsize*subgridsize*sizeof(complex<float>);
    auto size_spheroidal = 1ULL * subgridsize*subgridsize*sizeof(float);
    auto size_grid = 1ULL * nr_polarizations*gridsize*gridsize*sizeof(complex<float>);
    auto size_metadata = 1ULL * nr_subgrids*5*sizeof(int);
    auto size_subgrids = 1ULL * nr_subgrids*nr_polarizations*subgridsize*subgridsize*sizeof(complex<float>);

    // Allocate OpenCL buffers
    std::clog << ">>> Allocate OpenCL buffers" << std::endl;
    cl::Buffer h_visibilities = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, size_visibilities);
    cl::Buffer h_uvw          = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, size_uvw);
    cl::Buffer d_wavenumbers  = cl::Buffer(context, CL_MEM_READ_WRITE, size_wavenumbers);
    cl::Buffer d_aterm        = cl::Buffer(context, CL_MEM_READ_WRITE, size_aterm);
    cl::Buffer d_spheroidal   = cl::Buffer(context, CL_MEM_READ_WRITE, size_spheroidal);
    cl::Buffer h_grid         = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, size_grid);
    cl::Buffer h_metadata     = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, size_metadata);
    cl::Buffer h_subgrids     = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, size_subgrids);

    // Initialize data structures
    std::clog << ">>> Initialize data structures" << std::endl;
    void *wavenumbers  = idg::init_wavenumbers(nr_channels);
    void *aterm        = idg::init_aterm(nr_stations, nr_timeslots, nr_polarizations, subgridsize);
    void *spheroidal   = idg::init_spheroidal(subgridsize);
    void *visibilities = queue.enqueueMapBuffer(h_visibilities, CL_TRUE, CL_MAP_WRITE, 0, size_visibilities);
    void *uvw          = queue.enqueueMapBuffer(h_uvw, CL_TRUE, CL_MAP_WRITE, 0, size_uvw);
    void *metadata     = queue.enqueueMapBuffer(h_metadata, CL_TRUE, CL_MAP_WRITE, 0, size_metadata);
    idg::init_visibilities(visibilities, nr_baselines, nr_timesteps*nr_timeslots, nr_channels, nr_polarizations);
    idg::init_uvw(uvw, nr_stations, nr_baselines, nr_timesteps*nr_timeslots, gridsize, subgridsize);
    idg::init_metadata(metadata, uvw, wavenumbers, nr_stations, nr_baselines, nr_timesteps, nr_timeslots, nr_channels, gridsize, subgridsize, imagesize);
    queue.enqueueUnmapMemObject(h_visibilities, visibilities);
    queue.enqueueUnmapMemObject(h_uvw, uvw);
    queue.enqueueUnmapMemObject(h_metadata, metadata);
    queue.finish();

    // Initialize interface to kernels
    clog << ">>> Initialize proxy" << endl;
    idg::proxy::opencl::Reference opencl(params, context, deviceNumber);
    clog << endl;

    // Run gridder
    clog << ">>> Run gridder" << endl;
    opencl.grid_onto_subgrids(nr_subgrids, 0, h_uvw, d_wavenumbers, h_visibilities, d_spheroidal, d_aterm, h_metadata, h_subgrids);

    // Run degridder
    clog << ">>> Run degridder" << endl;
    opencl.degrid_from_subgrids(nr_subgrids, 0, h_uvw, d_wavenumbers, h_visibilities, d_spheroidal, d_aterm, h_metadata, h_subgrids);

    // Free memory for data structures
    free(wavenumbers);
    free(aterm);
    free(spheroidal);

    return EXIT_SUCCESS;
}
