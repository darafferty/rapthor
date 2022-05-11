// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef CU_WRAPPER_H
#define CU_WRAPPER_H

#include <string>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <nvToolsExt.h>

#include "common/auxiliary.h"
#include "idg-config.h"

struct dim3;

namespace cu {

void init(unsigned flags = 0);

template <typename T>
class Error : public std::exception {
 public:
  Error(T result, std::string& message) : _result(result), _message(message) {}

  const char* what() const throw() { return _message.c_str(); };

  operator T() const { return _result; }

 private:
  T _result;
  std::string _message;
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
      std::cerr << "CUDA Error: could not get attribute: " << attribute
                << std::endl;
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
  Context(Device& device, int flags = 0);
  ~Context();

 private:
  void setCurrent() const;
  void freeCurrent() const;
  CUcontext _context;
  CUdevice _device;

  friend class ScopedContext;
};

class ScopedContext {
 public:
  ScopedContext(const Context& context);
  ~ScopedContext();

 private:
  const Context& _context;
};

class HostMemory : public idg::auxiliary::Memory {
 public:
  HostMemory(const Context& context, size_t size = 0,
             int flags = CU_MEMHOSTALLOC_PORTABLE);
  ~HostMemory() override;

 private:
  void release();
  const Context& context_;
  int flags_;
};

class RegisteredMemory : public idg::auxiliary::Memory {
 public:
  RegisteredMemory(const Context& context, void* ptr, size_t size,
                   int flags = CU_MEMHOSTREGISTER_PORTABLE);
  ~RegisteredMemory() override;

 private:
  void release();
  const Context& context_;
  int flags_;
};

class DeviceMemory : public idg::auxiliary::Memory {
 public:
  DeviceMemory(const Context& context, size_t size);
  ~DeviceMemory();

  void zero() override;
  void zero(CUstream stream);

  operator CUdeviceptr() { return device_ptr_; }

 private:
  void release();
  const Context& context_;
  CUdeviceptr device_ptr_;
};

class UnifiedMemory : public idg::auxiliary::Memory {
 public:
  UnifiedMemory(const Context& context, size_t size,
                unsigned flags = CU_MEM_ATTACH_GLOBAL);
  ~UnifiedMemory() override;

  operator CUdeviceptr() { return reinterpret_cast<CUdeviceptr>(data()); }

  void set_advice(CUmem_advise advise);
  void set_advice(CUmem_advise advise, Device& device);

 private:
  void release();
  const Context& context_;
  int flags_ = 0;
};

class Source {
 public:
  Source(const char* input_file_name);

  void compile(const char* ptx_name, const char* compile_options = 0);

 private:
  const char* input_file_name;
};

class Module {
 public:
  Module(const Context& context, const char* file_name);
  Module(const Context& context, const void* data);
  ~Module();

  operator CUmodule();

 private:
  const Context& _context;
  CUmodule _module;
};

class Function {
 public:
  Function(const Context& context, Module& module, const char* name);
  Function(const Context& context, CUfunction function);

  int get_attribute(CUfunction_attribute attribute);
  void setCacheConfig(CUfunc_cache config);

  operator CUfunction();

 private:
  const Context& _context;
  CUfunction _function;
};

class Event {
 public:
  Event(const Context& context, int flags = CU_EVENT_DEFAULT);
  Event(const Event& event) = delete;
  Event(Event&& event);
  Event& operator=(const Event& event) = delete;
  Event& operator=(Event&& event);
  ~Event();

  void synchronize();
  float elapsedTime(Event& second);

  operator CUevent();

 private:
  const Context* _context;
  CUevent _event;
};

class Stream {
 public:
  Stream(const Context& context, int flags = CU_STREAM_DEFAULT);
  ~Stream();

  void memcpyHtoD(CUdeviceptr devPtr, const void* hostPtr, size_t size);
  void memcpyDtoH(void* hostPtr, CUdeviceptr devPtr, size_t size);
  void memcpyDtoD(CUdeviceptr dstPtr, CUdeviceptr srcPtr, size_t size);
  void memcpyHtoDAsync(CUdeviceptr devPtr, const void* hostPtr, size_t size);
  void memcpyDtoHAsync(void* hostPtr, CUdeviceptr devPtr, size_t size);
  void memcpyDtoDAsync(CUdeviceptr dstPtr, CUdeviceptr srcPtr, size_t size);
  void launchKernel(Function& function, unsigned gridX, unsigned gridY,
                    unsigned gridZ, unsigned blockX, unsigned blockY,
                    unsigned blockZ, unsigned sharedMemBytes,
                    const void** parameters);
  void launchKernel(Function& function, dim3 grid, dim3 block,
                    unsigned sharedMemBytes, const void** parameters);
  void query();
  void synchronize();
  void waitEvent(Event& event);
  void addCallback(CUstreamCallback callback, void* userData, int flags = 0);
  void record(Event& event);

  operator CUstream();

 private:
  const Context& _context;
  CUstream _stream;
};

class Marker {
 public:
  enum Color { red, green, blue, yellow, black };

  Marker(const char* message, unsigned color = 0xff00ff00);
  Marker(const char* message, Marker::Color color);
  void start();
  void end();

 private:
  unsigned int convert(Color color);
  nvtxEventAttributes_t _attributes;
  nvtxRangeId_t _id;
};

class Profiler {
 public:
  Profiler(const Context& context);
  ~Profiler();

 private:
  const Context& _context;
};

}  // end namespace cu

#endif
