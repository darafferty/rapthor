#ifndef CU_WRAPPER_H
#define CU_WRAPPER_H

#include <string>
#include <iostream>
#include <cuda.h>

struct dim3;

namespace cu {

    void init(unsigned flags = 0);

    class Error : public std::exception {
        public:
            Error(CUresult result);

            const char *what() const throw();

            operator CUresult() const;

        private:
            CUresult _result;
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
            //template <CUdevice_attribute attribute> int getAttribute() const;

            template <CUdevice_attribute attribute>
            int getAttribute() const {
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
            Context(Device device, int flags = 0);
            ~Context();
            void setCurrent() const;
            void setCacheConfig(CUfunc_cache config);
            void setSharedMemConfig(CUsharedconfig config);
            void synchronize();

            operator CUcontext();

        private:
            CUcontext _context;
    };


    class HostMemory {
        public:
            HostMemory(size_t size, int flags = 0);
            HostMemory(void *ptr, size_t size, int flags = 0);
            ~HostMemory();

            void update(void *ptr, size_t size);

            size_t size();
            void set(const void *in);
            void set(void *in);
            void set(void *in, size_t bytes);
            void zero();

            template <typename T> operator T *() {
                return static_cast<T *>(_ptr);
            }

        private:
            void *_ptr;
            size_t _size;
            bool free = false;
            bool unregister = false;
    };


    class DeviceMemory  {
        public:
            DeviceMemory(size_t size);
            DeviceMemory(void *ptr);
            ~DeviceMemory();

            size_t size();
            void set(void *in);
            void* get(size_t offset);
            void zero();

            template <typename T> operator T *() {
                return static_cast<T *>(&_ptr);
            }

            template <typename T> operator T () {
                return static_cast<T>(_ptr);
            }

        private:
            CUdeviceptr _ptr;
            size_t _size;
            bool free = false;
    };


    class Array {
        public:
            Array(unsigned width, CUarray_format format, unsigned numChannels);
            Array(unsigned width, unsigned height, CUarray_format format, unsigned numChannels);
            Array(unsigned width, unsigned height, unsigned depth, CUarray_format format, unsigned numChannels);
            ~Array();

            operator CUarray();

        private:
            CUarray _array;
    };


    class TexRef {
        public:
            TexRef(CUtexref texref);

            void setAddress(size_t &byte_offset, DeviceMemory &memory, size_t size);
            void setArray(Array &array, unsigned flags = CU_TRSA_OVERRIDE_FORMAT);
            void setAddressMode(int dim, CUaddress_mode am);
            void setFilterMode(CUfilter_mode fm);
            void setFlags(int flags);
            void setFormat(CUarray_format fmt, int numPackedComponents);

            operator CUtexref();

        private:
            CUtexref _texref;
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

            TexRef getTexRef(const char *name);

            operator CUmodule();

        private:
            CUmodule _module;
    };


    class Function {
        public:
            Function(Module &module, const char *name);

            int getAttribute(CUfunction_attribute attribute);
            void setCacheConfig(CUfunc_cache config);
            void paramSetTexRef(TexRef &texref);

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

} // end namespace cu

#endif
