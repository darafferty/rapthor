#include "Util.h"

float get_opencl_version(cl::Device &device) {
    const char *cl_device_version = device.getInfo<CL_DEVICE_VERSION>().c_str();
    char cl_version[3];
    strncpy(cl_version, cl_device_version + 7, 3);
    return atof(cl_version);
}

void printDevice(cl::Device &device, bool marker) {
    std::clog << "Device: "			  << device.getInfo<CL_DEVICE_NAME>();
    if (marker) {
        std::clog << "\t" << "<---";
    }
    std::clog << std::endl;
    std::clog << "Driver version  : " << device.getInfo<CL_DRIVER_VERSION>() << std::endl;
    std::clog << "Device version  : " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;
    std::clog << "Compute units   : " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
    std::clog << "Clock frequency : " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << " MHz" << std::endl;
    std::clog << "Global memory   : " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() * 1e-9 << " Gb" << std::endl;
    std::clog << "Local memory    : " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() * 1e-6 << " Mb" << std::endl;
    std::clog << std::endl;
}

void printDevices(int deviceNumber) {
    // Create context
    cl::Context context = cl::Context(CL_DEVICE_TYPE_ALL);

	// Get devices
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

	std::clog << "Devices" << std::endl;
	for (int d = 0; d < devices.size(); d++) {
        cl::Device &device = devices[d];
        bool marker = d == deviceNumber;
        printDevice(device, marker);
    }
	std::clog << "\n";
}
