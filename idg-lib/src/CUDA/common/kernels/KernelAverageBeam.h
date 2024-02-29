#ifndef IDG_CUDA_KERNEL_AVERAGE_BEAM_H_
#define IDG_CUDA_KERNEL_AVERAGE_BEAM_H_

#include <string>

#include "../KernelFactory.h"
#include "Kernel.h"

namespace idg::kernel::cuda {
class KernelAverageBeam : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelAverageBeam(cu::Device& device, cu::Stream& stream,
                    const cu::Module& module,
                    const Parameters& parameters = {});

  void enqueue(int nr_baselines, int nr_antennas, int nr_timesteps,
               int nr_channels, int nr_aterms, int subgrid_size,
               cu::DeviceMemory& d_uvw, cu::DeviceMemory& d_baselines,
               cu::DeviceMemory& d_aterms, cu::DeviceMemory& d_aterm_offsets,
               cu::DeviceMemory& d_weights, cu::DeviceMemory& d_average_beam);

  static constexpr unsigned kBlockSizeX = 128;
};

template <>
CompileDefinitions KernelFactory<KernelAverageBeam>::compileDefinitions() const;

}  // namespace idg::kernel::cuda

#endif  // IDG_CUDA_KERNEL_AVERAGE_BEAM_H_
