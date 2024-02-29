#ifndef IDG_CUDA_KERNEL_DEGRIDDER_H_
#define IDG_CUDA_KERNEL_DEGRIDDER_H_

#include <string>

#include "../KernelFactory.h"
#include "Kernel.h"

namespace idg::kernel::cuda {
class KernelDegridder : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelDegridder(cu::Device& device, cu::Stream& stream,
                  const cu::Module& module, const Parameters& parameters = {});

  void enqueue(int time_offset, int nr_subgrids, int nr_polarizations,
               int grid_size, int subgrid_size, float image_size, float w_step,
               int nr_channels, int nr_stations, float shift_l, float shift_m,
               cu::DeviceMemory& d_uvw, cu::DeviceMemory& d_wavenumbers,
               cu::DeviceMemory& d_visibilities, cu::DeviceMemory& d_taper,
               cu::DeviceMemory& d_aterms, cu::DeviceMemory& d_aterm_indices,
               cu::DeviceMemory& d_metadata, cu::DeviceMemory& d_subgrid);

  static constexpr unsigned kBlockSizeX = 64;
};

template <>
CompileDefinitions KernelFactory<KernelDegridder>::compileDefinitions() const;

}  // namespace idg::kernel::cuda

#endif  // IDG_CUDA_KERNEL_DEGRIDDER_H_
