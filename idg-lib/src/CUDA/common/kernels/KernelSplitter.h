#ifndef IDG_CUDA_KERNEL_SPLITTER_H_
#define IDG_CUDA_KERNEL_SPLITTER_H_

#include <string>

#include "../KernelFactory.h"
#include "Kernel.h"

namespace idg::kernel::cuda {
class KernelSplitter : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelSplitter(cu::Device& device, cu::Stream& stream,
                 const cu::Module& module, const Parameters& parameters = {});

  void enqueue(int nr_subgrids, int nr_polarizations, long grid_size,
               int subgrid_size, cu::DeviceMemory& d_metadata,
               cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_grid);

  static constexpr unsigned kBlockSizeX = 128;
};

template <>
CompileDefinitions KernelFactory<KernelSplitter>::compileDefinitions() const;

}  // namespace idg::kernel::cuda

#endif  // IDG_CUDA_KERNEL_SPLITTER_H_
