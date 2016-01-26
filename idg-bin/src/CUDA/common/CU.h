#include <string.h>

#if !defined CU_WRAPPER_H
#define CU_WRAPPER_H

#include <cuda.h>
#include <exception>
#include <string>
#include <iostream>

#define checkCudaCall(val)	__checkCudaCall((val), #val, __FILE__, __LINE__)

namespace cu {
	class Error : public std::exception {
		public:
		Error(CUresult result):
		_result(result)
		{}

		const char *what() const throw();

		operator CUresult () const {
			return _result;
		}

		private:
			CUresult _result;
	};


	#if 1
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
	#else
	inline void checkCudaCall(CUresult result) {
		if (result != CUDA_SUCCESS) {
			throw Error(result);
		}
	}
	#endif


	inline void init(unsigned flags = 0) {
		checkCudaCall(cuInit(flags));
	}


	class Device {
		public:
			static int getCount() {
				int nrDevices;
				checkCudaCall(cuDeviceGetCount(&nrDevices));
				return nrDevices;
			}

			Device() {}

			Device(int ordinal) {
				checkCudaCall(cuDeviceGet(&_device, ordinal));
			}

			std::string getName() const {
				char name[64];
				checkCudaCall(cuDeviceGetName(name, sizeof name, _device));
				return std::string(name);
			}

			template <CUdevice_attribute attribute> int getAttribute() const {
				int value;
				checkCudaCall(cuDeviceGetAttribute(&value, attribute, _device));
				return value;
			}

			operator CUdevice () {
				return _device;
			}

            size_t free_memory() {
                size_t free;
                size_t total;
                cuMemGetInfo(&free, &total);
                return free;
            }

            size_t total_memory() {
                size_t free;
                size_t total;
                cuMemGetInfo(&free, &total);
                return total;
            }

		private:
			CUdevice _device;
	};


	class Context {
		public:
            Context() {
                _context = NULL;
            }

			Context(Device device, int flags = 0) {
				checkCudaCall(cuCtxCreate(&_context, flags, device));
			}

			~Context() {
    				checkCudaCall(cuCtxDestroy(_context));
			}

			void setCurrent() const {
				checkCudaCall(cuCtxSetCurrent(_context));
			}

			void setCacheConfig(CUfunc_cache config) {
				checkCudaCall(cuCtxSetCacheConfig(config));
			}

			void setSharedMemConfig(CUsharedconfig config) {
				checkCudaCall(cuCtxSetSharedMemConfig(config));
			}

            void synchronize() {
                checkCudaCall(cuCtxSynchronize());
            }

			operator CUcontext () {
				return _context;
			}

		private:
			CUcontext _context;
	};


	class HostMemory {
		public:
			HostMemory(size_t size, int flags = 0) {
				_size = size;
				checkCudaCall(cuMemHostAlloc(&_ptr, size, flags));
                free = true;
			}

            HostMemory(void *ptr, size_t size, int flags= 0) {
                _size = size;
                _ptr = ptr;
                checkCudaCall(cuMemHostRegister(ptr, size, flags));
                unregister = true;
            }

        	~HostMemory() {
                if (free) {
				    checkCudaCall(cuMemFreeHost(_ptr));
                }
                if (unregister) {
                    cuMemHostUnregister(_ptr);
                }
			}

            template <typename T> operator T * () {
				return static_cast<T *>(_ptr);
			}

			size_t size() {
				return _size;
			}

            void set(void *in) {
                memcpy(_ptr, in, (size_t) _size);
            }

            void set(void *in, size_t bytes) {
                memcpy(_ptr, in, bytes);
            }

            void get(void *out) {
                memcpy(out, _ptr, (size_t) _size);
            }

			void zero() {
				memset(_ptr, 0, _size);
			}

		private:
			void *_ptr;
			size_t _size;
            bool free = false;
            bool unregister = false;
	};

	class DeviceMemory 	{
		public:
			DeviceMemory(size_t size) {
				_size = size;
				checkCudaCall(cuMemAlloc(&_ptr, size));
			}

			~DeviceMemory() {
				checkCudaCall(cuMemFree(_ptr));
			}

			operator CUdeviceptr() {
				return _ptr;
			}

			operator const void*() {
				return &_ptr;
			}

			size_t size() {
				return _size;
			}

            void set(void *in) {
                cuMemcpyHtoD(_ptr, in, _size);
            }

            void get(void *out) {
                cuMemcpyDtoH(out, _ptr, _size);
            }

			void zero() {
				cuMemsetD8(_ptr, 0, _size);
			}

		private:
			CUdeviceptr _ptr;
			size_t _size;

	};


	class Array {
		public:
			Array(unsigned width, CUarray_format format, unsigned numChannels) {
				Array(width, 0, format, numChannels);
			}

			Array(unsigned width, unsigned height, CUarray_format format, unsigned numChannels) {
				CUDA_ARRAY_DESCRIPTOR descriptor;
				descriptor.Width       = width;
				descriptor.Height      = height;
				descriptor.Format      = format;
				descriptor.NumChannels = numChannels;
				checkCudaCall(cuArrayCreate(&_array, &descriptor));
			}

			Array(unsigned width, unsigned height, unsigned depth, CUarray_format format, unsigned numChannels) {
				CUDA_ARRAY3D_DESCRIPTOR descriptor;
				descriptor.Width       = width;
				descriptor.Height      = height;
				descriptor.Depth       = depth;
				descriptor.Format      = format;
				descriptor.NumChannels = numChannels;
				descriptor.Flags       = 0;
				checkCudaCall(cuArray3DCreate(&_array, &descriptor));
			}

			~Array() {
				checkCudaCall(cuArrayDestroy(_array));
			}

			operator CUarray () {
				return _array;
			}

		private:
			CUarray _array;
	};


	class TexRef {
		public:
		TexRef(CUtexref texref):
			_texref(texref)
		{}

		void setAddress(size_t &byte_offset, DeviceMemory &memory, size_t size) {
			checkCudaCall(cuTexRefSetAddress(&byte_offset, _texref, memory, size));
		}

		void setArray(Array &array, unsigned flags = CU_TRSA_OVERRIDE_FORMAT) {
			checkCudaCall(cuTexRefSetArray(_texref, array, flags));
		}

		void setAddressMode(int dim, CUaddress_mode am) {
			checkCudaCall(cuTexRefSetAddressMode(_texref, dim, am));
		}

		void setFilterMode(CUfilter_mode fm) {
			checkCudaCall(cuTexRefSetFilterMode(_texref, fm));
		}

		void setFlags(int flags) {
			checkCudaCall(cuTexRefSetFlags(_texref, flags));
		}

		void setFormat(CUarray_format fmt, int numPackedComponents) {
			checkCudaCall(cuTexRefSetFormat(_texref, fmt, numPackedComponents));
		}

		operator CUtexref() {
			return _texref;
		}

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
			Module(const char *file_name) {
				checkCudaCall(cuModuleLoad(&_module, file_name));
			}

			Module(const void *data) {
				checkCudaCall(cuModuleLoadData(&_module, data));
			}

			~Module() {
				checkCudaCall(cuModuleUnload(_module));
			}

			TexRef getTexRef(const char *name) {
				CUtexref texref;
				checkCudaCall(cuModuleGetTexRef(&texref, _module, name));
				return TexRef(texref);
			}

			operator CUmodule () {
				return _module;
			}

		private:
			CUmodule _module;
	};


	class Function {
		public:
			Function(Module &module, const char *name) {
				checkCudaCall(cuModuleGetFunction(&_function, module, name));
			}

			int getAttribute(CUfunction_attribute attribute) {
				int value;
				checkCudaCall(cuFuncGetAttribute(&value, attribute, _function));
				return value;
			}

			void setCacheConfig(CUfunc_cache config) {
				checkCudaCall(cuFuncSetCacheConfig(_function, config));
			}

			void paramSetTexRef(TexRef &texref) {
				checkCudaCall(cuParamSetTexRef(_function, CU_PARAM_TR_DEFAULT, texref));
			}

			operator CUfunction () {
				return _function;
			}

		private:
			CUfunction _function;
	};


	class Event {
		public:
			Event(int flags = CU_EVENT_DEFAULT) {
				checkCudaCall(cuEventCreate(&_event, flags));
			}

			~Event() {
				checkCudaCall(cuEventDestroy(_event));
			}

			void synchronize() {
				checkCudaCall(cuEventSynchronize(_event));
			}

			float elapsedTime(Event &second) {
				float ms;
				checkCudaCall(cuEventElapsedTime(&ms, second, _event));
				return ms;
			}

			operator CUevent () {
				return _event;
			}

		private:
			CUevent _event;
	};


	class Stream {
		public:
			Stream(int flags = CU_STREAM_DEFAULT) {
				checkCudaCall(cuStreamCreate(&_stream, flags));
			}

			~Stream() {
				checkCudaCall(cuStreamDestroy(_stream));
			}

			void memcpyHtoDAsync(DeviceMemory &devPtr, const void *hostPtr) {
				checkCudaCall(cuMemcpyHtoDAsync(devPtr, hostPtr, devPtr.size(), _stream));
			}

			void memcpyHtoDAsync(CUdeviceptr devPtr, const void *hostPtr, size_t size) {
				checkCudaCall(cuMemcpyHtoDAsync(devPtr, hostPtr, size, _stream));
			}

			void memcpyDtoHAsync(void *hostPtr, DeviceMemory &devPtr) {
				checkCudaCall(cuMemcpyDtoHAsync(hostPtr, devPtr, devPtr.size(), _stream));
			}

			void memcpyDtoHAsync(void *hostPtr, CUdeviceptr devPtr, size_t size) {
				checkCudaCall(cuMemcpyDtoHAsync(hostPtr, devPtr, size, _stream));
			}

			void launchKernel(Function &function, unsigned gridX, unsigned gridY, unsigned gridZ, unsigned blockX, unsigned blockY, unsigned blockZ, unsigned sharedMemBytes, const void **parameters) {
				checkCudaCall(cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes, _stream, const_cast<void **>(parameters), 0));
			}

			void query() {
				checkCudaCall(cuStreamQuery(_stream));
			}

			void synchronize() {
				checkCudaCall(cuStreamSynchronize(_stream));
			}

			void waitEvent(Event &event) {
				checkCudaCall(cuStreamWaitEvent(_stream, event, 0));
			}

			void addCallback(CUstreamCallback callback, void *userData, int flags = 0) {
				checkCudaCall(cuStreamAddCallback(_stream, callback, userData, flags));
			}

			void record(Event &event) {
				checkCudaCall(cuEventRecord(event, _stream));
			}

			operator CUstream () {
				return _stream;
			}

		private:
			CUstream _stream;
	};
}

#endif
