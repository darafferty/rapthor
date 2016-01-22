#include <iostream>
#include <cstdlib>
#include <complex>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "idg-utility.h"  // Data init routines

using namespace std;

typedef struct {
    std::string deviceName;
    std::string driverVersion;

    unsigned numCUs;
    unsigned maxWGSize;
    unsigned maxAllocSize;
    unsigned maxGlobalSize;
    unsigned maxClockFreq;

    bool doubleSupported;
    cl_device_type  deviceType;

} device_info_t;

device_info_t getDeviceInfo(cl::Device &d) {
    device_info_t devInfo;

    devInfo.deviceName = d.getInfo<CL_DEVICE_NAME>();
    devInfo.driverVersion = d.getInfo<CL_DRIVER_VERSION>();

    devInfo.numCUs = (unsigned)d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    std::vector<size_t> maxWIPerDim;
    maxWIPerDim = d.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    devInfo.maxWGSize = (unsigned)maxWIPerDim[0];
    devInfo.maxAllocSize = (unsigned)d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    devInfo.maxGlobalSize = (unsigned)d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    devInfo.maxClockFreq = (unsigned)d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
    devInfo.doubleSupported = false;

    std::string extns = d.getInfo<CL_DEVICE_EXTENSIONS>();
    if ((extns.find("cl_khr_fp64") != std::string::npos) ||
    	(extns.find("cl_amd_fp64") != std::string::npos)) {
        devInfo.doubleSupported = true;
    }

    devInfo.deviceType = d.getInfo<CL_DEVICE_TYPE>();

    return devInfo;
}

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

template <typename PROXYNAME>
void run() {
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
    int nr_time = params.get_nr_time();
    int nr_timeslots = params.get_nr_timeslots();
    int nr_channels = params.get_nr_channels();
    int gridsize = params.get_grid_size();
    int subgridsize = params.get_subgrid_size();
    float imagesize = params.get_imagesize();
    int nr_polarizations = 4;
    int nr_subgrids = nr_baselines * nr_timeslots;

    float w_offset = 0;
    int kernel_size = (subgridsize / 4) + 1;

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
    auto size_visibilities = 1ULL * nr_baselines*nr_time*nr_channels*nr_polarizations*sizeof(complex<float>);
    auto size_uvw = 1ULL * nr_baselines*nr_time*3*sizeof(float);
    auto size_wavenumbers = 1ULL * nr_channels*sizeof(float);
    auto size_aterm = 1ULL * nr_stations*nr_timeslots*nr_polarizations*subgridsize*subgridsize*sizeof(complex<float>);
    auto size_spheroidal = 1ULL * subgridsize*subgridsize*sizeof(float);
    auto size_grid = 1ULL * nr_polarizations*gridsize*gridsize*sizeof(complex<float>);
    auto size_subgrids = 1ULL * nr_subgrids*nr_polarizations*subgridsize*subgridsize*sizeof(complex<float>);

    // Allocate OpenCL buffers
    std::clog << ">>> Allocate OpenCL buffers" << std::endl;
    cl::Buffer h_visibilities = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, size_visibilities);
    cl::Buffer h_uvw          = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, size_uvw);
    cl::Buffer d_wavenumbers  = cl::Buffer(context, CL_MEM_READ_WRITE, size_wavenumbers);
    cl::Buffer d_aterm        = cl::Buffer(context, CL_MEM_READ_WRITE, size_aterm);
    cl::Buffer d_spheroidal   = cl::Buffer(context, CL_MEM_READ_WRITE, size_spheroidal);
    cl::Buffer h_grid         = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, size_grid);
    cl::Buffer h_subgrids     = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, size_subgrids);

    // Initialize data structures
    std::clog << ">>> Initialize data structures" << std::endl;
    void *wavenumbers  = idg::init_wavenumbers(nr_channels);
    void *aterm        = idg::init_aterm(nr_stations, nr_timeslots, nr_polarizations, subgridsize);
    void *spheroidal   = idg::init_spheroidal(subgridsize);
    void *visibilities = queue.enqueueMapBuffer(h_visibilities, CL_FALSE, CL_MAP_WRITE, 0, size_visibilities);
    void *uvw          = queue.enqueueMapBuffer(h_uvw, CL_FALSE, CL_MAP_WRITE, 0, size_uvw);
    auto aterm_offsets = new int[nr_timeslots+1];
    queue.finish();
    idg::init_visibilities(visibilities, nr_baselines, nr_time, nr_channels, nr_polarizations);
    idg::init_uvw(uvw, nr_stations, nr_baselines, nr_time);
    idg::init_aterm_offsets(aterm_offsets, nr_timeslots, nr_time);
    queue.enqueueWriteBuffer(d_wavenumbers, CL_FALSE, 0, size_wavenumbers, wavenumbers);
    queue.enqueueWriteBuffer(d_aterm, CL_FALSE, 0, size_aterm, aterm);
    queue.enqueueWriteBuffer(d_spheroidal, CL_FALSE, 0, size_spheroidal, spheroidal);
    queue.enqueueUnmapMemObject(h_visibilities, visibilities);
    queue.enqueueUnmapMemObject(h_uvw, uvw);
    queue.finish();

    // Initialize interface to kernels
    clog << ">>> Initialize proxy" << endl;
    PROXYNAME opencl(params, context, deviceNumber);
    clog << endl;

    clog << ">>> Run fft" << endl;
    opencl.transform(idg::FourierDomainToImageDomain, h_grid);

    // Run gridder
    clog << ">>> Run gridder" << endl;
    opencl.grid_onto_subgrids(nr_subgrids, 0, h_uvw, d_wavenumbers, h_visibilities, d_spheroidal, d_aterm, aterm_offsets, h_subgrids);

    // Run degridder
    clog << ">>> Run degridder" << endl;
    opencl.degrid_from_subgrids(nr_subgrids, 0, h_uvw, d_wavenumbers, h_visibilities, d_spheroidal, d_aterm, aterm_offsets, h_subgrids);

    // Free memory for data structures
    free(wavenumbers);
    free(aterm);
    free(spheroidal);
}
