#include "Util.h"

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

void printDevice(cl::Device &device, bool marker) {
    std::clog << "Device: "			  << device.getInfo<CL_DEVICE_NAME>();
    if (marker) {
        std::clog << "\t" << "<---";
    }
    std::clog << std::endl;
    std::clog << "Driver version  : " << device.getInfo<CL_DRIVER_VERSION>() << std::endl;
    std::clog << "Compute units   : " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
    std::clog << "Clock frequency : " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << " MHz" << std::endl;
    std::clog << "Global memory   : " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() * 1e-9 << " Gb" << std::endl;
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
