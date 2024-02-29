#ifndef IDG_KERNEL_FACTORY_H_
#define IDG_KERNEL_FACTORY_H_

#include <memory>
#include <string>

#include "kernels/Kernel.h"

namespace idg::kernel::cuda {

// Abstract base class of the templated KernelFactory class.
class KernelFactoryBase {
 public:
  // Pure virtual destructor, because this is an abstract base class.
  virtual ~KernelFactoryBase() = 0;

 protected:
  // Return compile definitions to use when creating PTX code for any
  // Kernel.
  CompileDefinitions compileDefinitions(
      const Kernel::Parameters& parameters) const;

  // Return compile flags to use when creating PTX code for any Kernel.
  CompileFlags compileFlags(const Kernel::Parameters& param) const;
};

// Declaration of a generic factory class.
template <typename T>
class KernelFactory : public KernelFactoryBase {
 public:
  // Construct a factory for creating Kernel objects of type \c T, using the
  // settings provided by \a parameters.
  KernelFactory(cu::Device& device,
                const typename T::Parameters& parameters = {})
      : device_(device),
        parameters_(parameters),
        ptx_(createPTX(device, T::source_file_, compileDefinitions(),
                       compileFlags())) {}

  // Create a new Kernel object of type \c T.
  std::unique_ptr<T> create(const cu::Context& context,
                            cu::Stream& stream) const {
    return std::make_unique<T>(device_, stream,
                               createModule(context, T::source_file_, ptx_),
                               parameters_);
  }

 private:
  // Return compile definitions to use when creating PTX code for kernels of
  // type \c T, using the parameters stored in \c parameters_.
  CompileDefinitions compileDefinitions() const {
    return KernelFactoryBase::compileDefinitions(parameters_);
  }

  // Return compile flags to use when creating PTX code for kernels of type
  // \c T.
  CompileFlags compileFlags() const {
    return KernelFactoryBase::compileFlags(parameters_);
  }

  // The device to compiler kernels for
  cu::Device& device_;

  // Additional parameters needed to create a Kernel object of type \c T.
  typename T::Parameters parameters_;

  // PTX code, generated for kernels of type \c T, using information in the
  // Parset that was passed to the constructor.
  std::string ptx_;
};

}  // namespace idg::kernel::cuda

#endif  // IDG_KERNEL_FACTORY_H_
