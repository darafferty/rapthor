#include <complex>

#include <cudawrappers/cu.hpp>

#include "KernelScaler.h"

namespace idg::kernel::cuda {

std::string KernelScaler::source_file_ = "KernelScaler.cu";
std::string KernelScaler::kernel_function_ = "kernel_scaler";

KernelScaler::KernelScaler(cu::Device& device, cu::Stream& stream,
                           const cu::Module& module,
                           const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelScaler::enqueue(int nr_subgrids, int nr_polarizations,
                           int subgrid_size, cu::DeviceMemory& d_subgrid) {
  setArg(0, nr_polarizations);
  setArg(1, subgrid_size);
  setArg(2, d_subgrid);

  Grid grid(nr_subgrids);
  Block block(KernelScaler::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions KernelFactory<KernelScaler>::compileDefinitions() const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}
}  // namespace idg::kernel::cuda
