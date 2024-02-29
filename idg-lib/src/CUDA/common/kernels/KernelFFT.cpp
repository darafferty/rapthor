#include <complex>

#include <cudawrappers/cufft.hpp>

#include "KernelFFT.h"

namespace idg {

enum DomainAtoDomainB {
  FourierDomainToImageDomain,
  ImageDomainToFourierDomain
};

namespace kernel::cuda {

KernelFFT::KernelFFT(cu::Stream& stream, const Parameters& parameters)
    : Kernel(stream, parameters) {}

class KernelFFTImpl : public KernelFFT {
 public:
  KernelFFTImpl(cu::Stream& stream, const Parameters& parameters);

  virtual void enqueue(cu::DeviceMemory& data, int direction,
                       size_t n) override;

  virtual size_t batch() override { return parameters_.batch_; };

 private:
  Parameters parameters_;

  std::unique_ptr<cufft::FFT2D<CUDA_C_32F>> handle_;
};

std::unique_ptr<KernelFFT> KernelFFT::create(cu::Stream& stream,
                                             const Parameters& parameters) {
  return std::make_unique<KernelFFTImpl>(stream, parameters);
}

KernelFFTImpl::KernelFFTImpl(cu::Stream& stream, const Parameters& parameters)
    : KernelFFT(stream, parameters), parameters_(parameters) {
  const size_t size = parameters.size_;
  const size_t stride = 1;
  const size_t dist = size * size;
  const size_t batch = parameters.batch_;
  handle_ = std::make_unique<cufft::FFT2D<CUDA_C_32F>>(size, size, stride, dist,
                                                       batch);
  handle_->setStream(stream);
}

void KernelFFTImpl::enqueue(cu::DeviceMemory& data, int direction, size_t n) {
  const size_t batch = parameters_.batch_;
  const size_t size = parameters_.size_;

  cufftComplex* data_ptr =
      reinterpret_cast<cufftComplex*>(static_cast<CUdeviceptr>(data));
  direction =
      (direction == FourierDomainToImageDomain) ? CUFFT_INVERSE : CUFFT_FORWARD;

  // Execute fft in batches of size batch
  for (size_t s = 0; (s + batch) <= n; s += batch) {
    cu::DeviceMemory data(reinterpret_cast<CUdeviceptr>(data_ptr));
    handle_->execute(data, data, direction);
    data_ptr += size * size * batch;
  }

  // Execute remainder (if any)
  const size_t remainder = n % batch;
  if (remainder > 0) {
    const size_t sizeof_batch =
        batch * size * size * sizeof(std::complex<float>);
    cu::DeviceMemory d_temp = stream_.memAllocAsync(sizeof_batch);
    cu::DeviceMemory d_data_ptr(reinterpret_cast<CUdeviceptr>(data_ptr));
    const size_t sizeof_remainder =
        remainder * size * size * sizeof(std::complex<float>);
    stream_.memcpyDtoDAsync(d_temp, d_data_ptr, sizeof_remainder);
    handle_->execute(d_temp, d_temp, direction);
    stream_.memcpyDtoDAsync(d_data_ptr, d_temp, sizeof_remainder);
    stream_.memFreeAsync(d_temp);
  }
}

}  // namespace kernel::cuda
}  // namespace idg
