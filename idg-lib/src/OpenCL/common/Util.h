// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

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
void printDevices(unsigned deviceNumber);

#define BUFFER_BATCH_SIZE 1024 * 1024 * 1024  // 1 Gb

void writeBuffer(cl::CommandQueue &queue, cl::Buffer &buffer, cl_bool blocking,
                 const void *ptr);

void writeBufferBatched(cl::CommandQueue &queue, cl::Buffer &buffer,
                        cl_bool blocking, const void *ptr);

void readBuffer(cl::CommandQueue &queue, cl::Buffer &buffer, cl_bool blocking,
                void *ptr);

void zeroBuffer(cl::CommandQueue &queue, cl::Buffer &buffer);

void *mapBuffer(cl::CommandQueue &queue, cl::Buffer &buffer, cl_bool blocking,
                cl_map_flags flags);

void unmapBuffer(cl::CommandQueue &queue, cl::Buffer &buffer, const void *ptr);
#endif
