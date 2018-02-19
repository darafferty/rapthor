#include "CU.h"

#include <sstream>
#include <cstring>
#include <stdexcept>
#include <cassert>

#include <vector_types.h>

#define checkCudaCall(val)  __checkCudaCall((val), #val, __FILE__, __LINE__)

namespace cu {

    inline void __checkCudaCall(
        cudaError_t result,
        char const *const func,
        const char *const file,
        int const line)
    {
        if (result != cudaSuccess) {
            std::cerr << "CUDA Error at " << file;
            std::cerr << ":" << line;
            std::cerr << " in function " << func;
            std::cerr << ": " << cudaGetErrorString(result);
            std::cerr << std::endl;
            throw Error<cudaError_t>(result);
        }
    }

    inline void __checkCudaCall(
        CUresult result,
        char const *const func,
        const char *const file,
        int const line)
    {
        if (result != CUDA_SUCCESS) {
            const char *msg;
            cuGetErrorString(result, &msg);
            std::cerr << "CUDA Error at " << file;
            std::cerr << ":" << line;
            std::cerr << " in function " << func;
            std::cerr << ": " << msg;
            std::cerr << std::endl;
            throw Error<CUresult>(result);
        }
    }


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
        _capacity = size;
        _size = size;
        _flags = flags;
        register_memory(size, flags);
        free = true;
    }

    HostMemory::HostMemory(void *ptr, size_t size, int flags) {
        _capacity = size;
        _size = size;
        _flags = flags;
        assert(ptr != NULL);
        register_memory(size, flags, ptr);
        assert(_ptr == ptr);
        unregister = true;
    }

    void HostMemory::release() {
        if (free) {
            checkCudaCall(cuMemFreeHost(_ptr));
        }
        if (unregister) {
            checkCudaCall(cuMemHostUnregister(_ptr));
        }
    }

    HostMemory::~HostMemory() {
        release();
    }

    size_t HostMemory::capacity() {
        return _capacity;
    }

    size_t HostMemory::size() {
        return _size;
    }

    void HostMemory::resize(size_t size) {
        _size = size;
        if (size > _capacity) {
            release();

            if (free) {
                checkCudaCall(cuMemHostAlloc(&_ptr, size, _flags));
            }
            if (unregister) {
                checkCudaCall(cuMemHostRegister(_ptr, size, _flags));
            }
            _capacity = size;
        }
    }

    void HostMemory::zero() {
        memset(_ptr, 0, _size);
    }

    void* HostMemory::get(size_t offset) {
        return (void *) ((size_t) _ptr + offset);
    }

    std::vector<cu::HostMemory*> cu::HostMemory::registered_memory = std::vector<cu::HostMemory*>();

    void HostMemory::register_memory(
        uint64_t size,
        int flags,
        void* ptr)
    {
        bool register_memory = true;

        if (ptr == NULL) {
            // allocate new memory
            checkCudaCall(cuMemHostAlloc(&_ptr, size, flags));

            // cuMemHostAlloc already registers the memory
            register_memory = false;
        }

        // detect whether this pointer is managed
        bool managed;
        checkCudaCall(cuPointerGetAttribute(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, (CUdeviceptr) ptr));
        if (managed) {
            _ptr = ptr;
            register_memory = false;
        }

        // detect whether this pointer is already registered
        for (int i = 0; i < registered_memory.size(); i++) {
            HostMemory* m = registered_memory[i];
            auto *m_ptr = m->get();
            auto m_size = m->size();
            assert(m_ptr != NULL);

            // same pointer, smaller or equal size
            if (ptr == m_ptr && size <= m_size) {
                // the memory can safely be reused
                return;
            }

            // check pointer aliasing
            if ((((size_t) ptr + size) < (size_t) m_ptr) ||(size_t) ptr > ((size_t) m_ptr + m_size)) {
                // pointer outside of current memory
            } else {
                // overlap between current memory
                delete m;
                registered_memory.erase(registered_memory.begin() + i);
                i--;
            }
        }

        if (register_memory) {
            // register current memory
            _ptr = ptr;
            checkCudaCall(cuMemHostRegister(_ptr, size, flags));
        }

        // store the new memory
        registered_memory.push_back(this);
        assert(_ptr != NULL);
    }


    /*
        DeviceMemory
    */
    DeviceMemory::DeviceMemory(size_t size) {
        _capacity = size;
        _size = size;
        checkCudaCall(cuMemAlloc(&_ptr, size));
    }

    DeviceMemory::~DeviceMemory() {
        checkCudaCall(cuMemFree(_ptr));
    }

    size_t DeviceMemory::capacity() {
        return _capacity;
    }

    size_t DeviceMemory::size() {
        return _size;
    }

    void DeviceMemory::resize(size_t size) {
        _size = size;
        if (size > _capacity) {
            checkCudaCall(cuMemFree(_ptr));
            checkCudaCall(cuMemAlloc(&_ptr, size));
        }
    }

    void DeviceMemory::zero(CUstream stream) {
        if (stream != NULL) {
            cuMemsetD8Async(_ptr, 0, _size, stream);
        } else {
            cuMemsetD8(_ptr, 0, _size);
        }
    }

    /*
        UnifiedMemory
     */
    UnifiedMemory::UnifiedMemory(size_t size, unsigned flags) {
        _size = size;
        checkCudaCall(cudaMallocManaged(&_ptr, _size, flags));
    }

    UnifiedMemory::~UnifiedMemory() {
        checkCudaCall(cudaFree(_ptr));
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
        command_line << "nvcc -cubin ";
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
            throw cu::Error<CUresult>(CUDA_ERROR_INVALID_SOURCE);
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

    Function::Function(CUfunction function) {
        _function = function;
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
