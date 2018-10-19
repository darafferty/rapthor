#include <array>

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

void printDevices(unsigned deviceNumber) {
    // Create context
    cl::Context context = cl::Context(CL_DEVICE_TYPE_ALL);

	// Get devices
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

	std::clog << "Devices" << std::endl;
	for (unsigned d = 0; d < devices.size(); d++) {
        cl::Device &device = devices[d];
        bool marker = d == deviceNumber;
        printDevice(device, marker);
    }
	std::clog << "\n";
}

void writeBuffer(
        cl::CommandQueue &queue,
        cl::Buffer &buffer,
        cl_bool blocking,
        const void *ptr)
{
    size_t size   = buffer.getInfo<CL_MEM_SIZE>();
    size_t offset = 0;
    queue.enqueueWriteBuffer(buffer, blocking, offset, size, (void *) ptr);
}

void writeBufferBatched(
        cl::CommandQueue &queue,
        cl::Buffer &buffer,
        cl_bool blocking,
        const void *ptr)
{
    size_t size       = buffer.getInfo<CL_MEM_SIZE>();
    size_t batch_size = BUFFER_BATCH_SIZE;
    for (size_t offset_ = 0; offset_ < size; offset_ += batch_size) {
        size_t size_ = offset_ + batch_size > size ? size - offset_ : batch_size;
        const void *ptr_ = (const char *) ptr + offset_;
        queue.enqueueWriteBuffer(buffer, blocking, offset_, size_, ptr_);
    }
}

void readBuffer(
        cl::CommandQueue &queue,
        cl::Buffer &buffer,
        cl_bool blocking,
        void *ptr)
{
    size_t size   = buffer.getInfo<CL_MEM_SIZE>();
    size_t offset = 0;
    queue.enqueueReadBuffer(buffer, blocking, offset, size, ptr);
}

void zeroBuffer(
        cl::CommandQueue &queue,
        cl::Buffer &buffer)
{
    size_t offset    = 0;
    size_t size      = buffer.getInfo<CL_MEM_SIZE>();
    float pattern[1] = {0.0f};
    queue.enqueueFillBuffer<float[1]>(buffer, pattern, offset, size);
}

void* mapBuffer(
        cl::CommandQueue &queue,
        cl::Buffer &buffer,
        cl_bool blocking,
        cl_map_flags flags)
{
    size_t size      = buffer.getInfo<CL_MEM_SIZE>();
    size_t offset = 0;
    return queue.enqueueMapBuffer(buffer, blocking, flags, offset, size);
}

void unmapBuffer(
        cl::CommandQueue &queue,
        cl::Buffer &buffer,
        const void *ptr)
{
    queue.enqueueUnmapMemObject(buffer, (void *) ptr);
}
