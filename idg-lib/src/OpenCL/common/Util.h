#ifndef IDG_OPENCL_COMMON_H_
#define IDG_OPENCL_COMMON_H_

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <iostream>
#include <cstdlib>

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

device_info_t getDeviceInfo(cl::Device &d);

void printDevice(cl::Device &device, bool marker = false);
void printDevices(int deviceNumber);

#endif
