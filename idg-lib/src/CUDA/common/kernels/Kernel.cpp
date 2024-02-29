#include <iostream>
#include <string>

#include <cudawrappers/cu.hpp>

#include "Kernel.h"

namespace idg::kernel::cuda {

Kernel::Kernel(cu::Stream& stream, const Parameters& parameters)
    : stream_(stream), parameters_(parameters) {}

CompiledKernel::CompiledKernel(cu::Stream& stream,
                               std::unique_ptr<cu::Function> function,
                               const Parameters& parameters)
    : Kernel(stream, parameters), function_(std::move(function)) {
#if defined(DEBUG)
  std::cout << "Function " << function_->name() << ":"
            << "\n  nr. of registers used : "
            << function_->getAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS)
            << "\n  nr. of bytes of shared memory used (static) : "
            << function_->getAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
            << std::endl;
#endif
}

void CompiledKernel::setArg(size_t index, const cu::DeviceMemory& memory) {
  doSetArg(index, memory.parameter());
}

void CompiledKernel::doSetArg(size_t index, const void* argp) {
  if (index >= kernel_arguments_.size()) {
    kernel_arguments_.resize(index + 1);
  }
  kernel_arguments_[index] = argp;
}

void CompiledKernel::setEnqueueWorkSizes(Grid grid, Block block) {
  grid_ = grid;
  block_ = block;
}

void CompiledKernel::launch() {
  stream_.launchKernel(*function_, grid_.x, grid_.y, grid_.z, block_.x,
                       block_.y, block_.z, 0, kernel_arguments_);
}

}  // namespace idg::kernel::cuda
