#ifndef IDG_KERNEL_FFT_H_
#define IDG_KERNEL_FFT_H_

#include "Kernel.h"

namespace idg::kernel::cuda {

class KernelFFT : public Kernel {
 public:
  struct Parameters : public Kernel::Parameters {
    Parameters(size_t size, size_t batch) : size_(size), batch_(batch){};

    size_t size_;
    size_t batch_;
  };

  KernelFFT(cu::Stream& stream, const KernelFFT::Parameters& parameters);
  static std::unique_ptr<KernelFFT> create(cu::Stream& stream,
                                           const Parameters& parameters);
  virtual void enqueue(cu::DeviceMemory& data, int direction, size_t n) = 0;
  virtual size_t batch() = 0;
};

}  // namespace idg::kernel::cuda

#endif  // IDG_KERNEL_FFT_H_
