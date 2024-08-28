#include <complex>

#include <cudawrappers/cu.hpp>

#include "KernelCalibrate.h"

namespace idg::kernel::cuda {

/*
  LMNP
*/
std::string KernelCalibrateLMNP::source_file_ = "KernelCalibrate_lmnp.cu";
std::string KernelCalibrateLMNP::kernel_function_ = "kernel_calibrate_lmnp";

KernelCalibrateLMNP::KernelCalibrateLMNP(cu::Device& device, cu::Stream& stream,
                                         const cu::Module& module,
                                         const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelCalibrateLMNP::enqueue(int nr_subgrids, int grid_size,
                                  int subgrid_size, float image_size,
                                  float w_step, cu::DeviceMemory& d_metadata,
                                  cu::DeviceMemory& d_lmnp) {
  setArg(0, grid_size);
  setArg(1, subgrid_size);
  setArg(2, image_size);
  setArg(3, w_step);
  setArg(4, d_metadata);
  setArg(5, d_lmnp);

  Grid grid(nr_subgrids);
  Block block(KernelCalibrateLMNP::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions KernelFactory<KernelCalibrateLMNP>::compileDefinitions()
    const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}

/*
  Sums
*/
std::string KernelCalibrateSums::source_file_ = "KernelCalibrate_sums.cu";
std::string KernelCalibrateSums::kernel_function_ = "kernel_calibrate_sums";

KernelCalibrateSums::KernelCalibrateSums(cu::Device& device, cu::Stream& stream,
                                         const cu::Module& module,
                                         const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelCalibrateSums::enqueue(
    int nr_subgrids, int nr_polarizations, int subgrid_size, float image_size,
    int total_nr_timesteps, int nr_channels, int nr_stations, int term_offset,
    int current_nr_terms, int nr_terms, cu::DeviceMemory& d_uvw,
    cu::DeviceMemory& d_wavenumbers, cu::DeviceMemory& d_aterm,
    cu::DeviceMemory& d_aterm_derivatives, cu::DeviceMemory& d_aterm_indices,
    cu::DeviceMemory& d_metadata, cu::DeviceMemory& d_subgrid,
    cu::DeviceMemory& d_sums, cu::DeviceMemory& d_lmnp) {
  setArg(0, nr_polarizations);
  setArg(1, subgrid_size);
  setArg(2, image_size);
  setArg(3, total_nr_timesteps);
  setArg(4, nr_channels);
  setArg(5, nr_stations);
  setArg(6, term_offset);
  setArg(7, current_nr_terms);
  setArg(8, nr_terms);
  setArg(9, d_uvw);
  setArg(10, d_wavenumbers);
  setArg(11, d_aterm);
  setArg(12, d_aterm_derivatives);
  setArg(13, d_aterm_indices);
  setArg(14, d_metadata);
  setArg(15, d_subgrid);
  setArg(16, d_sums);
  setArg(17, d_lmnp);

  Grid grid(nr_subgrids);
  Block block(KernelCalibrateSums::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions KernelFactory<KernelCalibrateSums>::compileDefinitions()
    const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}

/*
  Gradient
*/
std::string KernelCalibrateGradient::source_file_ =
    "KernelCalibrate_gradient.cu";
std::string KernelCalibrateGradient::kernel_function_ =
    "kernel_calibrate_gradient";

KernelCalibrateGradient::KernelCalibrateGradient(cu::Device& device,
                                                 cu::Stream& stream,
                                                 const cu::Module& module,
                                                 const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelCalibrateGradient::enqueue(
    int nr_subgrids, int nr_polarizations, int subgrid_size, float image_size,
    int total_nr_timesteps, int nr_channels, int nr_stations, int term_offset,
    int current_nr_terms, int nr_terms, cu::DeviceMemory& d_uvw,
    cu::DeviceMemory& d_wavenumbers, cu::DeviceMemory& d_visibilities,
    cu::DeviceMemory& d_weights, cu::DeviceMemory& d_aterm,
    cu::DeviceMemory& d_aterm_indices, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_sums,
    cu::DeviceMemory& d_lmnp, cu::DeviceMemory& d_gradient,
    cu::DeviceMemory& d_residual) {
  setArg(0, nr_polarizations);
  setArg(1, subgrid_size);
  setArg(2, image_size);
  setArg(3, total_nr_timesteps);
  setArg(4, nr_channels);
  setArg(5, nr_stations);
  setArg(6, term_offset);
  setArg(7, current_nr_terms);
  setArg(8, nr_terms);
  setArg(9, d_uvw);
  setArg(10, d_wavenumbers);
  setArg(11, d_visibilities);
  setArg(12, d_weights);
  setArg(13, d_aterm);
  setArg(14, d_aterm_indices);
  setArg(15, d_metadata);
  setArg(16, d_subgrid);
  setArg(17, d_sums);
  setArg(18, d_lmnp);
  setArg(19, d_gradient);
  setArg(20, d_residual);

  Grid grid(nr_subgrids);
  Block block(KernelCalibrateSums::kBlockSizeX);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions KernelFactory<KernelCalibrateGradient>::compileDefinitions()
    const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}

/*
  Hessian
*/
std::string KernelCalibrateHessian::source_file_ = "KernelCalibrate_hessian.cu";
std::string KernelCalibrateHessian::kernel_function_ =
    "kernel_calibrate_hessian";

KernelCalibrateHessian::KernelCalibrateHessian(cu::Device& device,
                                               cu::Stream& stream,
                                               const cu::Module& module,
                                               const Parameters& parameters)
    : CompiledKernel(
          stream,
          std::make_unique<cu::Function>(module, kernel_function_.c_str()),
          parameters) {}

void KernelCalibrateHessian::enqueue(
    int nr_subgrids, int block_x, int block_y, int nr_polarizations,
    int total_nr_timesteps, int nr_channels, int term_offset_y,
    int term_offset_x, int nr_terms, cu::DeviceMemory& d_weights,
    cu::DeviceMemory& d_aterm_indices, cu::DeviceMemory& d_metadata,
    cu::DeviceMemory& d_sums_y, cu::DeviceMemory& d_sums_x,
    cu::DeviceMemory& d_hessian) {
  setArg(0, nr_polarizations);
  setArg(1, total_nr_timesteps);
  setArg(2, nr_channels);
  setArg(3, term_offset_y);
  setArg(4, term_offset_x);
  setArg(5, nr_terms);
  setArg(6, d_weights);
  setArg(7, d_aterm_indices);
  setArg(8, d_metadata);
  setArg(9, d_sums_y);
  setArg(10, d_sums_x);
  setArg(11, d_hessian);

  Grid grid(nr_subgrids);
  Block block(block_x, block_y);

  setEnqueueWorkSizes(grid, block);

  launch();
}

template <>
CompileDefinitions KernelFactory<KernelCalibrateHessian>::compileDefinitions()
    const {
  CompileDefinitions compile_definitions =
      KernelFactoryBase::compileDefinitions(parameters_);

  return compile_definitions;
}

}  // namespace idg::kernel::cuda
