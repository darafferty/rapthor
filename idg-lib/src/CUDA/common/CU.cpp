// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "CU.h"

#include <sstream>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <sstream>

#include <vector_types.h>

#define assertCudaCall(val) __assertCudaCall(val, #val, __FILE__, __LINE__)
#define checkCudaCall(val) __checkCudaCall(val, #val, __FILE__, __LINE__)

namespace cu {

/*
    Error checking
*/
inline void __assertCudaCall(CUresult result, char const *const func,
                             const char *const file, int const line) {
  if (result != CUDA_SUCCESS) {
    std::stringstream message_stream;
    const char *msg;
    cuGetErrorString(result, &msg);
    message_stream << "CUDA Error at " << file;
    message_stream << ":" << line;
    message_stream << " in function " << func;
    message_stream << ": " << msg;
    message_stream << std::endl;
    std::string message = message_stream.str();
    throw Error<CUresult>(result, message);
  }
}

inline void __checkCudaCall(CUresult result, char const *const func,
                            const char *const file, int const line) {
  try {
    __assertCudaCall(result, func, file, line);
  } catch (Error<CUresult> &error) {
    // pass
  }
}

/*
    Init
*/
void init(unsigned flags) { assertCudaCall(cuInit(flags)); }

/*
    Class Device
*/
int Device::getCount() {
  int nrDevices;
  assertCudaCall(cuDeviceGetCount(&nrDevices));
  return nrDevices;
}

Device::Device(int ordinal) { assertCudaCall(cuDeviceGet(&_device, ordinal)); }

std::string Device::get_name() const {
  char name[64];
  assertCudaCall(cuDeviceGetName(name, sizeof(name), _device));
  return std::string(name);
}

int Device::get_capability() const {
  int capability =
      10 * get_attribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>() +
      get_attribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();
  return capability;
}

Device::operator CUdevice() { return _device; }

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
    Context
*/
Context::Context(Device &device, int flags) {
  _device = device;
  assertCudaCall(cuCtxCreate(&_context, flags, device));

  // Make the context floating, so we can tie it to any thread
  freeCurrent();
}

Context::~Context() { assertCudaCall(cuCtxDestroy(_context)); }

void Context::setCurrent() const { assertCudaCall(cuCtxPushCurrent(_context)); }

void Context::freeCurrent() const { assertCudaCall(cuCtxPopCurrent(nullptr)); }

/*
    ScopedContext
*/
ScopedContext::ScopedContext(const Context &context) : _context(context) {
  _context.setCurrent();
}

ScopedContext::~ScopedContext() { _context.freeCurrent(); }

/*
    HostMemory
*/
HostMemory::HostMemory(const Context &context, size_t size, int flags)
    : _context(context) {
  ScopedContext scc(context);

  m_capacity = size;
  m_bytes = size;
  _flags = flags;
  if (size != 0) {
    void *ptr;
    assertCudaCall(cuMemHostAlloc(&ptr, size, _flags));
    set(ptr);
  }
}

HostMemory::~HostMemory() { release(); }

void HostMemory::resize(size_t size) {
  assert(size > 0);
  if (size < m_capacity) {
    m_bytes = size;
  } else if (size > m_capacity) {
    release();
    void *ptr;
    ScopedContext scc(_context);
    assertCudaCall(cuMemHostAlloc(&ptr, size, _flags));
    m_bytes = size;
    m_capacity = size;
    set(ptr);
  }
}

void HostMemory::release() {
  ScopedContext scc(_context);

  void *ptr = data();
  if (ptr) {
    assertCudaCall(cuMemFreeHost(ptr));
    set(nullptr);
  }
}

void HostMemory::zero() { memset(data(), 0, m_bytes); }

/*
    RegisteredMemory
*/
RegisteredMemory::RegisteredMemory(const Context &context, const void *ptr,
                                   size_t size, int flags)
    : _context(context) {
  ScopedContext scc(context);

  m_bytes = size;
  _flags = flags;
  assert(ptr != nullptr);
  set(const_cast<void *>(ptr));
  checkCudaCall(cuMemHostRegister(data(), size, _flags));
}

RegisteredMemory::~RegisteredMemory() { release(); }

void RegisteredMemory::resize(size_t size) {
  throw std::runtime_error("RegisteredMemory can not be resized!");
}

void RegisteredMemory::release() {
  ScopedContext scc(_context);
  checkCudaCall(cuMemHostUnregister(data()));
}

void RegisteredMemory::zero() { memset(data(), 0, m_bytes); }

/*
    DeviceMemory
*/

// Ensure the static const member _nullptr is not only declared but also defined
// otherwise it has no address

const CUdeviceptr DeviceMemory::_nullptr;

DeviceMemory::DeviceMemory(const Context &context, size_t size)
    : _context(context) {
  ScopedContext scc(_context);
  _capacity = size;
  _size = size;
  if (size) {
    assertCudaCall(cuMemAlloc(&_ptr, size));
  }
}

DeviceMemory::~DeviceMemory() { release(); }

size_t DeviceMemory::capacity() { return _capacity; }

size_t DeviceMemory::size() { return _size; }

void DeviceMemory::resize(size_t size) {
  ScopedContext scc(_context);
  _size = size;
  if (size > _capacity) {
    if (_capacity) assertCudaCall(cuMemFree(_ptr));
    assertCudaCall(cuMemAlloc(&_ptr, size));
    _capacity = size;
  }
}

void DeviceMemory::zero(CUstream stream) {
  ScopedContext scc(_context);
  if (_size) {
    if (stream != nullptr) {
      cuMemsetD8Async(_ptr, 0, _size, stream);
    } else {
      cuMemsetD8(_ptr, 0, _size);
    }
  }
}

void DeviceMemory::release() {
  ScopedContext scc(_context);
  if (_capacity) assertCudaCall(cuMemFree(_ptr));
}

/*
    UnifiedMemory
 */
UnifiedMemory::UnifiedMemory(const Context &context, void *ptr, size_t size)
    : _context(context) {
  ScopedContext scc(context);

  set(ptr);
  m_capacity = size;
  m_bytes = size;
}

UnifiedMemory::UnifiedMemory(const Context &context, size_t size,
                             unsigned flags)
    : _context(context) {
  ScopedContext scc(context);

  m_capacity = size;
  m_bytes = size;
  m_flags = flags;
  m_free = true;
  CUdeviceptr ptr;
  assertCudaCall(cuMemAllocManaged(&ptr, m_bytes, flags));
  set(reinterpret_cast<void *>(ptr));
}

UnifiedMemory::~UnifiedMemory() { release(); }

void UnifiedMemory::resize(size_t size) {
  assert(size > 0);
  if (size < m_capacity) {
    m_bytes = size;
  } else if (size > m_capacity) {
    release();
    ScopedContext scc(_context);
    CUdeviceptr ptr;
    assertCudaCall(cuMemAllocManaged(&ptr, size, m_flags));
    set(reinterpret_cast<void *>(ptr));
    m_capacity = size;
    m_bytes = size;
  }
}

void UnifiedMemory::set_advice(CUmem_advise advice) {
  CUdeviceptr ptr = reinterpret_cast<CUdeviceptr>(data());
  assertCudaCall(cuMemAdvise(ptr, m_bytes, advice, CU_DEVICE_CPU));
}

void UnifiedMemory::set_advice(CUmem_advise advice, Device &device) {
  CUdeviceptr ptr = reinterpret_cast<CUdeviceptr>(data());
  assertCudaCall(cuMemAdvise(ptr, m_bytes, advice, device));
}

void UnifiedMemory::release() {
  ScopedContext scc(_context);

  if (m_free) {
    CUdeviceptr ptr = reinterpret_cast<CUdeviceptr>(data());
    assertCudaCall(cuMemFree(ptr));
    m_free = false;
  }
}

/*
    Source
*/
Source::Source(const char *input_file_name)
    : input_file_name(input_file_name) {}

void Source::compile(const char *output_file_name,
                     const char *compiler_options) {
  std::stringstream command_line;
  command_line << NVCC;
  command_line << " -cubin ";
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
    std::string message(output_file_name);
    throw cu::Error<CUresult>(CUDA_ERROR_INVALID_SOURCE, message);
  }
}

/*
   Module
*/
Module::Module(const Context &context, const char *file_name)
    : _context(context) {
  ScopedContext scc(_context);

  assertCudaCall(cuModuleLoad(&_module, file_name));
}

Module::Module(const Context &context, const void *data) : _context(context) {
  ScopedContext scc(_context);

  assertCudaCall(cuModuleLoadData(&_module, data));
}

Module::~Module() {
  ScopedContext scc(_context);

  assertCudaCall(cuModuleUnload(_module));
}

Module::operator CUmodule() { return _module; }

/*
    Function
*/
Function::Function(const Context &context, Module &module, const char *name)
    : _context(context) {
  assertCudaCall(cuModuleGetFunction(&_function, module, name));
}

Function::Function(const Context &context, CUfunction function)
    : _context(context) {
  _function = function;
}

int Function::get_attribute(CUfunction_attribute attribute) {
  int value;
  assertCudaCall(cuFuncGetAttribute(&value, attribute, _function));
  return value;
}

void Function::setCacheConfig(CUfunc_cache config) {
  assertCudaCall(cuFuncSetCacheConfig(_function, config));
}

Function::operator CUfunction() { return _function; }

/*
    Event
*/
Event::Event(const Context &context, int flags) : _context(&context) {
  ScopedContext scc(*_context);
  assertCudaCall(cuEventCreate(&_event, flags));
}

Event::Event(Event &&event) : _context(event._context), _event(event._event) {
  event._context = nullptr;
}

Event &Event::operator=(Event &&event) {
  _context = event._context;
  _event = event._event;
  event._context = nullptr;
  return *this;
}

Event::~Event() {
  if (_context) {
    ScopedContext scc(*_context);
    assertCudaCall(cuEventDestroy(_event));
  }
}

void Event::synchronize() {
  if (_context) {
    ScopedContext scc(*_context);
    assertCudaCall(cuEventSynchronize(_event));
  }
}

float Event::elapsedTime(Event &second) {
  if (_context) {
    ScopedContext scc(*_context);
    float ms;
    assertCudaCall(cuEventElapsedTime(&ms, second, _event));
    return ms;
  } else {
    return 0;
  }
}

Event::operator CUevent() { return _event; }

/*
    Stream
*/
Stream::Stream(const Context &context, int flags) : _context(context) {
  ScopedContext scc(_context);
  assertCudaCall(cuStreamCreate(&_stream, flags));
}

Stream::~Stream() {
  ScopedContext scc(_context);
  assertCudaCall(cuStreamDestroy(_stream));
}

void Stream::memcpyHtoDAsync(CUdeviceptr devPtr, const void *hostPtr,
                             size_t size) {
  ScopedContext scc(_context);
  assertCudaCall(cuMemcpyHtoDAsync(devPtr, hostPtr, size, _stream));
}

void Stream::memcpyDtoHAsync(void *hostPtr, CUdeviceptr devPtr, size_t size) {
  ScopedContext scc(_context);
  assertCudaCall(cuMemcpyDtoHAsync(hostPtr, devPtr, size, _stream));
}

void Stream::memcpyDtoDAsync(CUdeviceptr dstPtr, CUdeviceptr srcPtr,
                             size_t size) {
  ScopedContext scc(_context);
  assertCudaCall(cuMemcpyDtoDAsync(dstPtr, srcPtr, size, _stream));
}

void Stream::launchKernel(Function &function, unsigned gridX, unsigned gridY,
                          unsigned gridZ, unsigned blockX, unsigned blockY,
                          unsigned blockZ, unsigned sharedMemBytes,
                          const void **parameters) {
  ScopedContext scc(_context);
  assertCudaCall(cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY,
                                blockZ, sharedMemBytes, _stream,
                                const_cast<void **>(parameters), 0));
}

void Stream::launchKernel(Function &function, dim3 grid, dim3 block,
                          unsigned sharedMemBytes, const void **parameters) {
  ScopedContext scc(_context);
  assertCudaCall(cuLaunchKernel(function, grid.x, grid.y, grid.z, block.x,
                                block.y, block.z, sharedMemBytes, _stream,
                                const_cast<void **>(parameters), 0));
}

void Stream::query() {
  ScopedContext scc(_context);
  assertCudaCall(cuStreamQuery(_stream));
}

void Stream::synchronize() {
  ScopedContext scc(_context);
  assertCudaCall(cuStreamSynchronize(_stream));
}

void Stream::waitEvent(Event &event) {
  ScopedContext scc(_context);
  assertCudaCall(cuStreamWaitEvent(_stream, event, 0));
}

void Stream::addCallback(CUstreamCallback callback, void *userData, int flags) {
  ScopedContext scc(_context);
  assertCudaCall(cuStreamAddCallback(_stream, callback, userData, flags));
}

void Stream::record(Event &event) {
  ScopedContext scc(_context);
  assertCudaCall(cuEventRecord(event, _stream));
}

Stream::operator CUstream() { return _stream; }

Marker::Marker(const char *message, unsigned color) {
  _attributes = {0};
  _attributes.version = NVTX_VERSION;
  _attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  _attributes.colorType = NVTX_COLOR_ARGB;
  _attributes.color = color;
  _attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
  _attributes.message.ascii = message;
}

Marker::Marker(const char *message, Color color)
    : Marker(message, convert(color)) {}

void Marker::start() { _id = nvtxRangeStartEx(&_attributes); }

void Marker::end() { nvtxRangeEnd(_id); }

unsigned int Marker::convert(Color color) {
  switch (color) {
    case red:
      return 0xffff0000;
    case green:
      return 0xff00ff00;
    case blue:
      return 0xff0000ff;
    case yellow:
      return 0xffffff00;
    case black:
      return 0xff000000;
    default:
      return 0xff00ff00;
  }
}

}  // end namespace cu
