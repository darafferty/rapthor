#include <complex>

#include <cudawrappers/cu.hpp>

#include "KernelAverageBeam.h"

namespace idg::kernel::cuda {

std::string KernelAverageBeam::source_file_ = "KernelAverageBeam.cu";
std::string KernelAverageBeam::kernel_function_ = "kernel_average_beam";

KernelAverageBeam::KernelAverageBeam(cu::Device& device, cu::Stream& stream,
                                     const cu::Module& module,
                                     const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelAverageBeam::enqueue(
    int nr_baselines, int nr_antennas, int nr_timesteps, int nr_channels,
    int nr_aterms, int subgrid_size, cu::DeviceMemory& d_uvw,
    cu::DeviceMemory& d_baselines, cu::DeviceMemory& d_aterms,
    cu::DeviceMemory& d_aterm_offsets, cu::DeviceMemory& d_weights,
    cu::DeviceMemory& d_average_beam) {
  setArg(0, nr_antennas);
  setArg(1, nr_timesteps);
  setArg(2, nr_channels);
  setArg(3, nr_aterms);
  setArg(4, subgrid_size);
  setArg(5, d_uvw);
  setArg(6, d_baselines);
  setArg(7, d_aterms);
  setArg(8, d_aterm_offsets);
  setArg(9, d_weights);
  setArg(10, d_average_beam);

  Grid grid(nr_baselines);
  Block block(KernelAverageBeam::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions KernelFactory<KernelAverageBeam>::compileDefinitions()
    const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}
}  // namespace idg::kernel::cuda
