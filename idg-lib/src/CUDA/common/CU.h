#ifndef CU_WRAPPER_H
#define CU_WRAPPER_H

#include <string>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <nvToolsExt.h>

struct dim3;

namespace cu {

    void init(unsigned flags = 0);

    template<typename T>
    class Error : public std::exception {
        public:
            Error(T result):
            _result(result) {}

            operator T() const {
                return _result;
            }

        private:
            T _result;
    };


    class Device {
        public:
            Device();
            Device(int ordinal);

            static int getCount();
            std::string get_name() const;
            int get_capability() const;
            size_t get_free_memory() const;
            size_t get_total_memory() const;

            template <CUdevice_attribute attribute>
            int get_attribute() const {
                int value;
                if (cuDeviceGetAttribute(&value, attribute, _device) != CUDA_SUCCESS) {
                    std::cerr << "CUDA Error: could not get attribute: " << attribute << std::endl;
                    exit(EXIT_FAILURE);
                }
                return value;
            }

            operator CUdevice();

        private:
            CUdevice _device;
    };


    class Context {
        public:
            Context();
            Context(Device& device, int flags = 0);
            ~Context();
            void setCurrent() const;
            void setCacheConfig(CUfunc_cache config);
            void setSharedMemConfig(CUsharedconfig config);
            void synchronize();
            void reset();

            operator CUcontext();

        private:
            CUcontext _context;
            CUdevice _device;
    };


    class HostMemory {
        public:
            HostMemory(size_t size, int flags = 0);
            HostMemory(void *ptr, size_t size, int flags = 0, bool register_memory = true);
            ~HostMemory();

            size_t capacity();
            size_t size();
            void resize(size_t size);
            void* get(size_t offset = 0);
            void zero();

            template <typename T> operator T *() {
                return static_cast<T *>(_ptr);
            }

        private:
            void release();
            void *_ptr;
            size_t _capacity;
            size_t _size;
            int _flags;
            bool allocated = false;
            bool registered = false;
    };


    class DeviceMemory  {

        public:
            DeviceMemory(size_t size);
            ~DeviceMemory();

            size_t capacity();
            size_t size();
            void resize(size_t size);
            void zero(CUstream stream = NULL);

            template <typename T> operator T *() {
                if (_size)
                {
                    return static_cast<T *>(&_ptr);
                }
                else
                {
                    return static_cast<T *>(&_nullptr);
                }
            }

            template <typename T> operator T () {
                if (_size)
                {
                    return static_cast<T>(_ptr);
                }
                else
                {
                    return static_cast<T>(_nullptr);
                }
            }

        private:
            CUdeviceptr _ptr;
            size_t _capacity;
            size_t _size;
            static const CUdeviceptr _nullptr = 0;
    };


    class UnifiedMemory {
        public:
            UnifiedMemory(void* ptr, size_t size);
            UnifiedMemory(size_t size, unsigned flags = CU_MEM_ATTACH_GLOBAL);
            ~UnifiedMemory();

            void* ptr() { return (void *) _ptr; }
            void set_advice(CUmem_advise advise);
            void set_advice(CUmem_advise advise, Device& device);

        private:
            CUdeviceptr _ptr;
            size_t _size;
            bool free = false;
    };


    class Source {
        public:
            Source(const char *input_file_name);

            void compile(const char *ptx_name, const char *compile_options = 0);

        private:
            const char *input_file_name;
    };


    class Module {
        public:
            Module(const char *file_name);
            Module(const void *data);
            ~Module();

            operator CUmodule();

        private:
            CUmodule _module;
    };


    class Function {
        public:
            Function(Module &module, const char *name);
            Function(CUfunction function);

            int get_attribute(CUfunction_attribute attribute);
            void setCacheConfig(CUfunc_cache config);

            operator CUfunction();

        private:
            CUfunction _function;
    };


    class Event {
        public:
            Event(int flags = CU_EVENT_DEFAULT);
            ~Event();

            void synchronize();
            float elapsedTime(Event &second);

            operator CUevent();

        private:
            CUevent _event;
    };


    class Stream {
        public:
            Stream(int flags = CU_STREAM_DEFAULT);
            ~Stream();

            void memcpyHtoDAsync(DeviceMemory &devPtr, const void *hostPtr);
            void memcpyHtoDAsync(CUdeviceptr devPtr, const void *hostPtr, size_t size);
            void memcpyDtoHAsync(void *hostPtr, DeviceMemory &devPtr);
            void memcpyDtoHAsync(void *hostPtr, CUdeviceptr devPtr, size_t size);
            void launchKernel(Function &function, unsigned gridX, unsigned gridY, unsigned gridZ, unsigned blockX, unsigned blockY, unsigned blockZ, unsigned sharedMemBytes, const void **parameters);
            void launchKernel(Function &function, dim3 grid, dim3 block, unsigned sharedMemBytes, const void **parameters);
            void query();
            void synchronize();
            void waitEvent(Event &event);
            void addCallback(CUstreamCallback callback, void *userData, int flags = 0);
            void record(Event &event);

            operator CUstream();

        private:
            CUstream _stream;
    };

    class Marker {
        public:
            Marker(
                const char *message,
                unsigned color = 0xff00ff00);
            void start();
            void end();

        private:
            nvtxEventAttributes_t _attributes;
            nvtxRangeId_t _id;
    };

} // end namespace cu

#endif
