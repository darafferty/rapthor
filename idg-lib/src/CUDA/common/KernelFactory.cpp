#include "KernelFactory.h"

#include "idg-common.h"

namespace idg::kernel::cuda {
KernelFactoryBase::~KernelFactoryBase() {}

CompileDefinitions KernelFactoryBase::compileDefinitions(
    const Kernel::Parameters& parameters) const {
  CompileDefinitions defs;
  return defs;
}

CompileFlags KernelFactoryBase::compileFlags(
    const Kernel::Parameters& parameters) const {
  CompileFlags flags;
  flags.insert("-I" + auxiliary::get_inc_dir());
  return flags;
}

}  // namespace idg::kernel::cuda
