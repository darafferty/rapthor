#include <complex>

#include <cudawrappers/cu.hpp>

#include "KernelFFTShift.h"

namespace idg::kernel::cuda {

std::string KernelFFTShift::source_file_ = "KernelFFTShift.cu";
std::string KernelFFTShift::kernel_function_ = "kernel_fft_shift";

KernelFFTShift::KernelFFTShift(cu::Device& device, cu::Stream& stream,
                               const cu::Module& module,
                               const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelFFTShift::enqueue(cu::DeviceMemory& d_data, size_t size,
                             size_t batch, std::complex<float>& scale) {
  setArg(0, size);
  setArg(1, d_data);
  setArg(2, scale);

  Grid grid(batch, ceil(size / 2.0));
  Block block(KernelFFTShift::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions KernelFactory<KernelFFTShift>::compileDefinitions() const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}
}  // namespace idg::kernel::cuda
