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

void printDevices(int deviceNumber) {
    // Create context
    cl::Context context = cl::Context(CL_DEVICE_TYPE_ALL);

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

    // Retrieve constants for memory allocation
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

    // Allocate and initialize data structures
    std::clog << ">>> Initialize data structures" << std::endl;

    auto size_visibilities = 1ULL * nr_baselines*nr_time*
        nr_channels*nr_polarizations;
    auto size_uvw = 1ULL * nr_baselines*nr_time*3;
    auto size_wavenumbers = 1ULL * nr_channels;
    auto size_aterm = 1ULL * nr_stations*nr_timeslots*
        nr_polarizations*subgridsize*subgridsize;
    auto size_spheroidal = 1ULL * subgridsize*subgridsize;
    auto size_grid = 1ULL * nr_polarizations*gridsize*gridsize;
    auto size_baselines = 1ULL * nr_baselines*2;

    auto visibilities = new std::complex<float>[size_visibilities];
    auto uvw = new float[size_uvw];
    auto wavenumbers = new float[size_wavenumbers];
    auto aterm = new std::complex<float>[size_aterm];
    auto aterm_offsets = new int[nr_timeslots+1];
    auto spheroidal = new float[size_spheroidal];
    auto grid = new std::complex<float>[size_grid];
    auto baselines = new int[size_baselines];

    idg::init_visibilities(visibilities, nr_baselines,
                           nr_time,
                           nr_channels, nr_polarizations);
    idg::init_uvw(uvw, nr_stations, nr_baselines, nr_time);
    idg::init_wavenumbers(wavenumbers, nr_channels);
    idg::init_aterm(aterm, nr_stations, nr_timeslots, nr_polarizations,
                    subgridsize);
    idg::init_aterm_offsets(aterm_offsets, nr_timeslots, nr_time);
    idg::init_spheroidal(spheroidal, subgridsize);
    idg::init_grid(grid, gridsize, nr_polarizations);
    idg::init_baselines(baselines, nr_stations, nr_baselines);
    std::clog << std::endl;

    // Initialize interface to kernels
    clog << ">>> Initialize proxy" << endl;
    PROXYNAME proxy(params, deviceNumber);
    clog << endl;

    // Print all OpenCL devices
    printDevices(deviceNumber);

    // Run
    clog << ">>> Run fft" << endl;
    proxy.transform(idg::FourierDomainToImageDomain, grid);

    clog << ">>> Run gridder" << endl;
    proxy.grid_visibilities(visibilities, uvw, wavenumbers, baselines, grid, w_offset, kernel_size, aterm, aterm_offsets, spheroidal);

    clog << ">>> Run fft" << endl;
    proxy.transform(idg::FourierDomainToImageDomain, grid);

    clog << ">>> Run degridder" << endl;
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
