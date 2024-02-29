#include <complex>

#include <cudawrappers/cu.hpp>

#include "KernelSplitter.h"

namespace idg::kernel::cuda {

std::string KernelSplitter::source_file_ = "KernelSplitter.cu";
std::string KernelSplitter::kernel_function_ = "kernel_splitter";

KernelSplitter::KernelSplitter(cu::Device& device, cu::Stream& stream,
                               const cu::Module& module,
                               const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelSplitter::enqueue(int nr_subgrids, int nr_polarizations,
                             long grid_size, int subgrid_size,
                             cu::DeviceMemory& d_metadata,
                             cu::DeviceMemory& d_subgrid,
                             cu::DeviceMemory& d_grid) {
  setArg(0, nr_polarizations);
  setArg(1, grid_size);
  setArg(2, subgrid_size);
  setArg(3, d_metadata);
  setArg(4, d_subgrid);
  setArg(5, d_grid);

  Grid grid(nr_subgrids);
  Block block(KernelSplitter::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions KernelFactory<KernelSplitter>::compileDefinitions() const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}
}  // namespace idg::kernel::cuda
