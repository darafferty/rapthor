#ifndef IDG_OPENCL_COMMON_H_
#define IDG_OPENCL_COMMON_H_

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <iostream>
#include <cstdlib>

using namespace std;

float get_opencl_version(cl::Device &device);
void printDevice(cl::Device &device, bool marker = false);
void printDevices(int deviceNumber);

#endif
