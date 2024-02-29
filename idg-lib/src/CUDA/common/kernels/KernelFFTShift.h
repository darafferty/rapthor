#ifndef IDG_CUDA_KERNEL_FFT_SHIFT_H_
#define IDG_CUDA_KERNEL_FFT_SHIFT_H_

#include <complex>
#include <string>

#include "../KernelFactory.h"
#include "Kernel.h"

namespace idg::kernel::cuda {
class KernelFFTShift : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelFFTShift(cu::Device& device, cu::Stream& stream,
                 const cu::Module& module, const Parameters& parameters = {});

  void enqueue(cu::DeviceMemory& d_data, size_t size, size_t batch,
               std::complex<float>& scale);

  static constexpr unsigned kBlockSizeX = 128;
};

template <>
CompileDefinitions KernelFactory<KernelFFTShift>::compileDefinitions() const;

}  // namespace idg::kernel::cuda

#endif  // IDG_CUDA_KERNEL_FFT_SHIFT_H_
