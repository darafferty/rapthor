#include "CU.h"

#include <sstream>
#include <cstring>

#include <cuda.h>
#include <vector_types.h>

#define checkCudaCall(val)  __checkCudaCall((val), #val, __FILE__, __LINE__)

namespace cu {
    /*
        Error
    */
    Error::Error(CUresult result) :
        _result(result) {};

    Error::operator CUresult() const {
        return _result;
    }


    inline void __checkCudaCall(CUresult result, char const *const func, const char *const file, int const line) {
        if (result != CUDA_SUCCESS) {
            std::cerr << "CUDA Error at " << file;
            std::cerr << ":" << line;
            std::cerr << " in function " << func;
            std::cerr << ": " << Error(result).what();
            std::cerr << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    const char *Error::what() const throw() {
        switch (_result) {
            case CUDA_SUCCESS:
                return "success";
            case CUDA_ERROR_INVALID_VALUE:
                return "invalid value";
            case CUDA_ERROR_OUT_OF_MEMORY:
                return "out of memory";
            case CUDA_ERROR_NOT_INITIALIZED:
                return "not initialized";
            case CUDA_ERROR_DEINITIALIZED:
                return "deinitialized";
            case CUDA_ERROR_PROFILER_DISABLED:
                return "profiler disabled";
            case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
                return "profiler not initialized";
            case CUDA_ERROR_PROFILER_ALREADY_STARTED:
                return "profiler already started";
            case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
                return "profiler already stopped";
            case CUDA_ERROR_NO_DEVICE:
                return "no device";
            case CUDA_ERROR_INVALID_DEVICE:
                return "invalid device";
            case CUDA_ERROR_INVALID_IMAGE:
                return "invalid image";
            case CUDA_ERROR_INVALID_CONTEXT:
                return "invalid context";
            case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
                return "context already current";
            case CUDA_ERROR_MAP_FAILED:
                return "map failed";
            case CUDA_ERROR_UNMAP_FAILED:
                return "unmap failed";
            case CUDA_ERROR_ARRAY_IS_MAPPED:
                return "array is mapped";
            case CUDA_ERROR_ALREADY_MAPPED:
                return "already mapped";
            case CUDA_ERROR_NO_BINARY_FOR_GPU:
                return "no binary for GPU";
            case CUDA_ERROR_ALREADY_ACQUIRED:
                return "already acquired";
            case CUDA_ERROR_NOT_MAPPED:
                return "not mapped";
            case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
                return "not mapped as array";
            case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
                return "not mapped as pointer";
            case CUDA_ERROR_ECC_UNCORRECTABLE:
                return "ECC uncorrectable";
            case CUDA_ERROR_UNSUPPORTED_LIMIT:
                return "unsupported limit";
            case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
                return "context already in use";
            case CUDA_ERROR_INVALID_SOURCE:
                return "invalid source";
            case CUDA_ERROR_FILE_NOT_FOUND:
                return "file not found";
            case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
                return "shared object symbol not found";
            case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
                return "shared object init failed";
            case CUDA_ERROR_OPERATING_SYSTEM:
                return "operating system";
            case CUDA_ERROR_INVALID_HANDLE:
                return "invalid handle";
            case CUDA_ERROR_NOT_FOUND:
                return "not found";
            case CUDA_ERROR_NOT_READY:
                return "not ready";
            case CUDA_ERROR_LAUNCH_FAILED:
                return "launch failed";
            case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
                return "launch out of resources";
            case CUDA_ERROR_LAUNCH_TIMEOUT:
                return "launch timeout";
            case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
                return "launch incompatible texturing";
            case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
                return "peer access already enabled";
            case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
                return "peer access not enabled";
            case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
                return "primary context active";
            case CUDA_ERROR_CONTEXT_IS_DESTROYED:
                return "context is destroyed";
            case CUDA_ERROR_UNKNOWN:
                return "unknown";
        default:
                return "unknown error code";
        }
    };


    /*
        Init
    */
    void init(unsigned flags) {
        checkCudaCall(cuInit(flags));
    }


    /*
        Class Device
    */
    int Device::getCount() {
        int nrDevices;
        checkCudaCall(cuDeviceGetCount(&nrDevices));
        return nrDevices;
    }

    Device::Device(int ordinal) {
        checkCudaCall(cuDeviceGet(&_device, ordinal));
    }

    std::string Device::get_name() const {
        char name[64];
        checkCudaCall(cuDeviceGetName(name, sizeof(name), _device));
        return std::string(name);
    }

    int Device::get_capability() const {
        int capability = 10 * getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>() +
                              getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();
        return capability;
    }

    Device::operator CUdevice() {
        return _device;
    }

    size_t Device::get_free_memory() const {
        size_t free;
        size_t total;
        cuMemGetInfo(&free, &total);
        return free;
    }

    size_t Device::get_total_memory() const {
        size_t free;
        size_t total;
        cuMemGetInfo(&free, &total);
        return total;
    }


    /*
        Class Context
    */
    Context::Context() {
        _context = NULL;
    }

    Context::Context(Device device, int flags) {
        checkCudaCall(cuCtxCreate(&_context, flags, device));
    }

    Context::~Context() {
        checkCudaCall(cuCtxDestroy(_context));
    }

    void Context::setCurrent() const {
        checkCudaCall(cuCtxSetCurrent(_context));
    }

    void Context::setCacheConfig(CUfunc_cache config) {
        checkCudaCall(cuCtxSetCacheConfig(config));
    }

    void Context::setSharedMemConfig(CUsharedconfig config) {
        checkCudaCall(cuCtxSetSharedMemConfig(config));
    }

    void Context::synchronize() {
        checkCudaCall(cuCtxSynchronize());
    }

    Context::operator CUcontext() {
        return _context;
    }


    /*
        HostMemory
    */
    HostMemory::HostMemory(size_t size, int flags) {
        _size = size;
        checkCudaCall(cuMemHostAlloc(&_ptr, size, flags));
        free = true;
    }

    HostMemory::HostMemory(void *ptr, size_t size, int flags) {
        _size = size;
        _ptr = ptr;
        checkCudaCall(cuMemHostRegister(ptr, size, flags));
        unregister = true;
    }

    HostMemory::~HostMemory() {
        if (free) {
            checkCudaCall(cuMemFreeHost(_ptr));
        }
        if (unregister) {
            cuMemHostUnregister(_ptr);
        }
    }

    size_t HostMemory::size() {
        return _size;
    }

    void HostMemory::set(const void *in) {
        memcpy(_ptr, in, (size_t) _size);
    }

    void HostMemory::set(void *in) {
        memcpy(_ptr, in, (size_t) _size);
    }

    void HostMemory::set(void *in, size_t bytes) {
        memcpy(_ptr, in, bytes);
    }

    void HostMemory::zero() {
        memset(_ptr, 0, _size);
    }


    /*
        DeviceMemory
    */
    DeviceMemory::DeviceMemory(size_t size) {
        _size = size;
        checkCudaCall(cuMemAlloc(&_ptr, size));
        free = true;
    }

    DeviceMemory::DeviceMemory(void *ptr) {
        cuMemHostGetDevicePointer(&_ptr, ptr, 0);
    }

    DeviceMemory::~DeviceMemory() {
        if (free) {
            checkCudaCall(cuMemFree(_ptr));
        }
    }

    DeviceMemory::operator CUdeviceptr() {
        return _ptr;
    }

    DeviceMemory::operator const void*() {
        return &_ptr;
    }

    size_t DeviceMemory::size() {
        return _size;
    }

    void DeviceMemory::set(void *in) {
        cuMemcpyHtoD(_ptr, in, _size);
    }

    void* DeviceMemory::get(size_t offset) {
        return (void *) (_ptr + offset);
    }

    void DeviceMemory::zero() {
        cuMemsetD8(_ptr, 0, _size);
    }


    /*
        Array
    */
    Array::Array(unsigned width, CUarray_format format, unsigned numChannels) {
        Array(width, 0, format, numChannels);
    }

    Array::Array(unsigned width, unsigned height, CUarray_format format, unsigned numChannels) {
        CUDA_ARRAY_DESCRIPTOR descriptor;
        descriptor.Width       = width;
        descriptor.Height      = height;
        descriptor.Format      = format;
        descriptor.NumChannels = numChannels;
        checkCudaCall(cuArrayCreate(&_array, &descriptor));
    }

    Array::Array(unsigned width, unsigned height, unsigned depth, CUarray_format format, unsigned numChannels) {
        CUDA_ARRAY3D_DESCRIPTOR descriptor;
        descriptor.Width       = width;
        descriptor.Height      = height;
        descriptor.Depth       = depth;
        descriptor.Format      = format;
        descriptor.NumChannels = numChannels;
        descriptor.Flags       = 0;
        checkCudaCall(cuArray3DCreate(&_array, &descriptor));
    }

    Array::~Array() {
        checkCudaCall(cuArrayDestroy(_array));
    }

    Array::operator CUarray() {
        return _array;
    }


    /*
        TexRef
    */
    TexRef::TexRef(CUtexref texref):
        _texref(texref)
    {}

    void TexRef::setAddress(size_t &byte_offset, DeviceMemory &memory, size_t size) {
        checkCudaCall(cuTexRefSetAddress(&byte_offset, _texref, memory, size));
    }

    void TexRef::setArray(Array &array, unsigned flags) {
        checkCudaCall(cuTexRefSetArray(_texref, array, flags));
    }

    void TexRef::setAddressMode(int dim, CUaddress_mode am) {
        checkCudaCall(cuTexRefSetAddressMode(_texref, dim, am));
    }

    void TexRef::setFilterMode(CUfilter_mode fm) {
        checkCudaCall(cuTexRefSetFilterMode(_texref, fm));
    }

    void TexRef::setFlags(int flags) {
        checkCudaCall(cuTexRefSetFlags(_texref, flags));
    }

    void TexRef::setFormat(CUarray_format fmt, int numPackedComponents) {
        checkCudaCall(cuTexRefSetFormat(_texref, fmt, numPackedComponents));
    }

    TexRef::operator CUtexref() {
        return _texref;
    }


    /*
        Source
    */

    Source::Source(const char *input_file_name):
        input_file_name(input_file_name)
        {}


    void Source::compile(const char *output_file_name, const char *compiler_options) {
        std::stringstream command_line;
        command_line << "nvcc -ptx ";
        command_line << compiler_options;
        command_line << " -o ";
        command_line << output_file_name;
        command_line << ' ' << input_file_name;

        #if defined(DEBUG)
        #pragma omp critical(cout)
        std::clog << "Compiling " << output_file_name << std::endl;
        std::clog << command_line.str() << std::endl;
        #endif
        int retval = system(command_line.str().c_str());

        if (WEXITSTATUS(retval) != 0) {
            throw cu::Error(CUDA_ERROR_INVALID_SOURCE);
        }
    }


    /*
       Module
    */
    Module::Module(const char *file_name) {
        checkCudaCall(cuModuleLoad(&_module, file_name));
    }

    Module::Module(const void *data) {
        checkCudaCall(cuModuleLoadData(&_module, data));
    }

    Module::~Module() {
        checkCudaCall(cuModuleUnload(_module));
    }

    TexRef Module::getTexRef(const char *name) {
        CUtexref texref;
        checkCudaCall(cuModuleGetTexRef(&texref, _module, name));
        return TexRef(texref);
    }

    Module::operator CUmodule() {
        return _module;
    }


    /*
        Function
    */
    Function::Function(Module &module, const char *name) {
        checkCudaCall(cuModuleGetFunction(&_function, module, name));
    }

    int Function::getAttribute(CUfunction_attribute attribute) {
        int value;
        checkCudaCall(cuFuncGetAttribute(&value, attribute, _function));
        return value;
    }

    void Function::setCacheConfig(CUfunc_cache config) {
        checkCudaCall(cuFuncSetCacheConfig(_function, config));
    }

    void Function::paramSetTexRef(TexRef &texref) {
        checkCudaCall(cuParamSetTexRef(_function, CU_PARAM_TR_DEFAULT, texref));
    }

    Function::operator CUfunction() {
        return _function;
    }


    /*
        Event
    */
        Event::Event(int flags) {
            checkCudaCall(cuEventCreate(&_event, flags));
        }

        Event::~Event() {
            checkCudaCall(cuEventDestroy(_event));
        }

        void Event::synchronize() {
            checkCudaCall(cuEventSynchronize(_event));
        }

        float Event::elapsedTime(Event &second) {
            float ms;
            checkCudaCall(cuEventElapsedTime(&ms, second, _event));
            return ms;
        }

        Event::operator CUevent() {
            return _event;
        }


    /*
        Stream
    */
    Stream::Stream(int flags) {
        checkCudaCall(cuStreamCreate(&_stream, flags));
    }

    Stream::~Stream() {
        checkCudaCall(cuStreamDestroy(_stream));
    }

    void Stream::memcpyHtoDAsync(DeviceMemory &devPtr, const void *hostPtr) {
        checkCudaCall(cuMemcpyHtoDAsync(devPtr, hostPtr, devPtr.size(), _stream));
    }

    void Stream::memcpyHtoDAsync(CUdeviceptr devPtr, const void *hostPtr, size_t size) {
        checkCudaCall(cuMemcpyHtoDAsync(devPtr, hostPtr, size, _stream));
    }

    void Stream::memcpyDtoHAsync(void *hostPtr, DeviceMemory &devPtr) {
        checkCudaCall(cuMemcpyDtoHAsync(hostPtr, devPtr, devPtr.size(), _stream));
    }

    void Stream::memcpyDtoHAsync(void *hostPtr, CUdeviceptr devPtr, size_t size) {
        checkCudaCall(cuMemcpyDtoHAsync(hostPtr, devPtr, size, _stream));
    }

    void Stream::memcpyHtoHAsync(HostMemory &dstPtr, const void *srcPtr) {
        checkCudaCall(cuMemcpyAsync((CUdeviceptr) (void *) dstPtr, (CUdeviceptr) srcPtr, dstPtr.size(), _stream));
    }

    void Stream::launchKernel(Function &function, unsigned gridX, unsigned gridY, unsigned gridZ, unsigned blockX, unsigned blockY, unsigned blockZ, unsigned sharedMemBytes, const void **parameters) {
        checkCudaCall(cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes, _stream, const_cast<void **>(parameters), 0));
    }

    void Stream::launchKernel(Function &function, dim3 grid, dim3 block, unsigned sharedMemBytes, const void **parameters) {
        checkCudaCall(cuLaunchKernel(function, grid.x, grid.y, grid.z, block.x, block.y, block.z, sharedMemBytes, _stream, const_cast<void **>(parameters), 0));
    }

    void Stream::query() {
        checkCudaCall(cuStreamQuery(_stream));
    }

    void Stream::synchronize() {
        checkCudaCall(cuStreamSynchronize(_stream));
    }

    void Stream::waitEvent(Event &event) {
        checkCudaCall(cuStreamWaitEvent(_stream, event, 0));
    }

    void Stream::addCallback(CUstreamCallback callback, void *userData, int flags) {
        checkCudaCall(cuStreamAddCallback(_stream, callback, userData, flags));
    }

    void Stream::record(Event &event) {
        checkCudaCall(cuEventRecord(event, _stream));
    }

    Stream::operator CUstream() {
        return _stream;
    }

} // end namespace cu
