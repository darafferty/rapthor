#ifndef IDG_CUDA_KERNEL_H_
#define IDG_CUDA_KERNEL_H_

#include <memory>
#include <vector>

#include "../gpu_utils.h"

namespace cu {
class DeviceMemory;
class Function;
class Stream;
}  // namespace cu

namespace idg::kernel::cuda {

class Kernel {
 public:
  struct Parameters {};

 protected:
  Kernel(cu::Stream& stream, const Parameters& parameters);

  cu::Stream& stream_;

  const Parameters& parameters_;
};

class CompiledKernel : public Kernel {
 protected:
  CompiledKernel(cu::Stream& stream, std::unique_ptr<cu::Function> function,
                 const Parameters& parameters);

  void setArg(size_t index, const cu::DeviceMemory& memory);

  template <typename T>
  void setArg(size_t index, const T& val) {
    doSetArg(index, &val);
  }

 protected:
  void setEnqueueWorkSizes(Grid grid, Block block);

  void launch();

 private:
  std::unique_ptr<cu::Function> function_;

  std::vector<const void*> kernel_arguments_;

  void doSetArg(size_t index, const void* argp);

  Grid grid_;

  Block block_;
};

}  // namespace idg::kernel::cuda

#endif  // IDG_CUDA_KERNEL_H_
