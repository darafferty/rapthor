#include <complex>

#include <boost/lexical_cast.hpp>
#include <cudawrappers/cu.hpp>

#include "KernelGridder.h"

namespace idg::kernel::cuda {

std::string KernelGridder::source_file_ = "KernelGridder.cu";
std::string KernelGridder::kernel_function_ = "kernel_gridder";

KernelGridder::KernelGridder(cu::Device& device, cu::Stream& stream,
                             const cu::Module& module,
                             const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters),
      parameters_(parameters) {}

void KernelGridder::enqueue(
    int time_offset, int nr_subgrids, int nr_polarizations, int grid_size,
    int subgrid_size, float image_size, float w_step, int nr_channels,
    int nr_stations, float shift_l, float shift_m, cu::DeviceMemory& d_uvw,
    cu::DeviceMemory& d_wavenumbers, cu::DeviceMemory& d_visibilities,
    cu::DeviceMemory& d_taper, cu::DeviceMemory& d_aterms,
    cu::DeviceMemory& d_aterm_indices, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_avg_aterm, cu::DeviceMemory& d_subgrid) {
  setArg(0, time_offset);
  setArg(1, nr_polarizations);
  setArg(2, grid_size);
  setArg(3, subgrid_size);
  setArg(4, image_size);
  setArg(5, w_step);
  setArg(6, shift_l);
  setArg(7, shift_m);
  setArg(8, nr_channels);
  setArg(9, nr_stations);
  setArg(10, d_uvw);
  setArg(11, d_wavenumbers);
  setArg(12, d_visibilities);
  setArg(13, d_taper);
  setArg(14, d_aterms);
  setArg(15, d_aterm_indices);
  setArg(16, d_metadata);
  int idx = 17;
  if (parameters_.enable_avg_aterm_) {
    setArg(idx++, d_avg_aterm);
  }
  setArg(idx, d_subgrid);

  Grid grid(nr_subgrids);
  Block block(KernelGridder::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions KernelFactory<KernelGridder>::compileDefinitions() const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  compile_definitions["BLOCK_SIZE_X"] =
      boost::lexical_cast<std::string>(KernelGridder::kBlockSizeX);

  compile_definitions["ENABLE_AVG_ATERM"] =
      boost::lexical_cast<std::string>(parameters_.enable_avg_aterm_);

#if defined(USE_EXTRAPOLATE)
  compile_definitions["USE_EXTRAPOLATE"] =
      boost::lexical_cast<std::string>(true);
#endif

  return compile_definitions;
}
}  // namespace idg::kernel::cuda
