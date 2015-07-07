#include <CL/cl.hpp>

#ifndef MAX
#define MAX(X, Y)       \
    (X > Y)? X: Y;
#endif
#ifndef MIN
#define MIN(X, Y)       \
    (X < Y)? X: Y;
#endif

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

    // Limiting max work-group size to 256
    #define MAX_WG_SIZE 256
    devInfo.maxWGSize = MIN(devInfo.maxWGSize, MAX_WG_SIZE);

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
