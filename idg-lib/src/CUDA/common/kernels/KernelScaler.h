#ifndef IDG_CUDA_KERNEL_SCALER_H_
#define IDG_CUDA_KERNEL_SCALER_H_

#include <string>

#include "../KernelFactory.h"
#include "Kernel.h"

namespace idg::kernel::cuda {
class KernelScaler : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelScaler(cu::Device& device, cu::Stream& stream, const cu::Module& module,
               const Parameters& parameters = {});

  void enqueue(int nr_subgrids, int nr_polarizations, int subgrid_size,
               cu::DeviceMemory& d_subgrid);

  static constexpr unsigned kBlockSizeX = 128;
};

template <>
CompileDefinitions KernelFactory<KernelScaler>::compileDefinitions() const;

}  // namespace idg::kernel::cuda

#endif  // IDG_CUDA_KERNEL_SCALER_H_
