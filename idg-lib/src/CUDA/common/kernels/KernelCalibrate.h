#ifndef IDG_CUDA_KERNEL_CALIBRATE_H_
#define IDG_CUDA_KERNEL_CALIBRATE_H_

#include <string>

#include "../KernelFactory.h"
#include "Kernel.h"

namespace idg::kernel::cuda {

/*
  LMNP
*/
class KernelCalibrateLMNP : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelCalibrateLMNP(cu::Device& device, cu::Stream& stream,
                      const cu::Module& module,
                      const Parameters& parameters = {});

  void enqueue(int nr_subgrids, int grid_size, int subgrid_size,
               float image_size, float w_step, cu::DeviceMemory& d_metadata,
               cu::DeviceMemory& d_lmnp);

  static constexpr unsigned kBlockSizeX = 128;
};

template <>
CompileDefinitions KernelFactory<KernelCalibrateLMNP>::compileDefinitions()
    const;

/*
  Sums
*/
class KernelCalibrateSums : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelCalibrateSums(cu::Device& device, cu::Stream& stream,
                      const cu::Module& module,
                      const Parameters& parameters = {});

  void enqueue(int nr_subgrids, int nr_polarizations, int subgrid_size,
               float image_size, int total_nr_timesteps, int nr_channels,
               int nr_stations, int term_offset, int current_nr_terms,
               int nr_terms, cu::DeviceMemory& d_uvw,
               cu::DeviceMemory& d_wavenumbers, cu::DeviceMemory& d_aterm,
               cu::DeviceMemory& d_aterm_derivatives,
               cu::DeviceMemory& d_aterm_indices, cu::DeviceMemory& d_metadata,
               cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_sums,
               cu::DeviceMemory& d_lmnp);

  static constexpr unsigned kBlockSizeX = 128;
};

template <>
CompileDefinitions KernelFactory<KernelCalibrateSums>::compileDefinitions()
    const;

/*
  Gradient
*/
class KernelCalibrateGradient : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelCalibrateGradient(cu::Device& device, cu::Stream& stream,
                          const cu::Module& module,
                          const Parameters& parameters = {});

  void enqueue(int nr_subgrids, int nr_polarizations, int subgrid_size,
               float image_size, int total_nr_timesteps, int nr_channels,
               int nr_stations, int term_offset, int current_nr_terms,
               int nr_terms, cu::DeviceMemory& d_uvw,
               cu::DeviceMemory& d_wavenumbers,
               cu::DeviceMemory& d_visibilities, cu::DeviceMemory& d_weights,
               cu::DeviceMemory& d_aterm, cu::DeviceMemory& d_aterm_indices,
               cu::DeviceMemory& d_metadata, cu::DeviceMemory& d_subgrid,
               cu::DeviceMemory& d_sums, cu::DeviceMemory& d_lmnp,
               cu::DeviceMemory& d_gradient, cu::DeviceMemory& d_residual);

  static constexpr unsigned kBlockSizeX = 128;
};

template <>
CompileDefinitions KernelFactory<KernelCalibrateGradient>::compileDefinitions()
    const;

/*
  Hessian
*/
class KernelCalibrateHessian : public CompiledKernel {
 public:
  static std::string source_file_;
  static std::string kernel_function_;

  KernelCalibrateHessian(cu::Device& device, cu::Stream& stream,
                         const cu::Module& module,
                         const Parameters& parameters = {});

  void enqueue(int nr_subgrids, int block_x, int block_y, int nr_polarizations,
               int total_nr_timesteps, int nr_channels, int term_offset_y,
               int term_offset_x, int nr_terms, cu::DeviceMemory& d_weights,
               cu::DeviceMemory& d_aterm_indices, cu::DeviceMemory& d_metadata,
               cu::DeviceMemory& d_sums_y, cu::DeviceMemory& d_sums_x,
               cu::DeviceMemory& d_hessian);
};

template <>
CompileDefinitions KernelFactory<KernelCalibrateHessian>::compileDefinitions()
    const;

}  // namespace idg::kernel::cuda

#endif  // IDG_CUDA_KERNEL_CALIBRATE_H_
