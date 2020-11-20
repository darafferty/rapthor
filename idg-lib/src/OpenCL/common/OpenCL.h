// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_OPENCLNEW_H_
#define IDG_OPENCLNEW_H_

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include "idg-common.h"

namespace cl {
class Context;
}

namespace idg {
namespace kernel {
namespace opencl {
class InstanceOpenCL;
}
}  // namespace kernel

namespace proxy {
namespace opencl {
class OpenCL : public Proxy {
 public:
  OpenCL();

  ~OpenCL();

 public:
  void print_compiler_flags();

  void print_devices();

  cl::Context& get_context() { return *context; }

  unsigned int get_num_devices() const;
  idg::kernel::opencl::InstanceOpenCL& get_device(unsigned int i) const;

  std::vector<int> compute_jobsize(const Plan& plan,
                                   const unsigned int nr_timesteps,
                                   const unsigned int nr_channels,
                                   const unsigned int subgrid_size,
                                   const unsigned int nr_streams);

 protected:
  void init_devices();
  void free_devices();

 private:
  cl::Context* context;
  std::vector<idg::kernel::opencl::InstanceOpenCL*> devices;
};
}  // namespace opencl
}  // end namespace proxy
}  // end namespace idg
#endif
