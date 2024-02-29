// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_INSTANCE_CUDA_H_
#define IDG_INSTANCE_CUDA_H_

#include <memory>

#include "KernelFactory.h"
#include "kernels/KernelAdder.h"
#include "kernels/KernelAverageBeam.h"
#include "kernels/KernelCalibrate.h"
#include "kernels/KernelDegridder.h"
#include "kernels/KernelFFT.h"
#include "kernels/KernelFFTShift.h"
#include "kernels/KernelGridder.h"
#include "kernels/KernelScaler.h"
#include "kernels/KernelSplitter.h"
#include "kernels/KernelWTiling.h"
#include "PowerRecord.h"

namespace idg::kernel::cuda {

class InstanceCUDA : public KernelsInstance {
 public:
  static constexpr size_t kFftBatch = 1024;

  InstanceCUDA(size_t device_id = 0);

  ~InstanceCUDA();

  cu::Context& get_context() const { return *context_; }
  cu::Device& get_device() const { return *device_; }
  cu::Stream& get_execute_stream() const { return *stream_execute_; };
  cu::Stream& get_htod_stream() const { return *stream_htod_; };
  cu::Stream& get_dtoh_stream() const { return *stream_dtoh_; };

  pmt::State measure();
  void measure(PowerRecord& record, cu::Stream& stream);

  void launch_gridder(
      int time_offset, int nr_subgrids, int nr_polarizations, int grid_size,
      int subgrid_size, float image_size, float w_step, int nr_channels,
      int nr_stations, float shift_l, float shift_m, cu::DeviceMemory& d_uvw,
      cu::DeviceMemory& d_wavenumbers, cu::DeviceMemory& d_visibilities,
      cu::DeviceMemory& d_taper, cu::DeviceMemory& d_aterms,
      cu::DeviceMemory& d_aterm_indices, cu::DeviceMemory& d_metadata,
      cu::DeviceMemory& d_avg_aterm, cu::DeviceMemory& d_subgrid);

  void launch_degridder(int time_offset, int nr_subgrids, int nr_polarizations,
                        int grid_size, int subgrid_size, float image_size,
                        float w_step, int nr_channels, int nr_stations,
                        float shift_l, float shift_m, cu::DeviceMemory& d_uvw,
                        cu::DeviceMemory& d_wavenumbers,
                        cu::DeviceMemory& d_visibilities,
                        cu::DeviceMemory& d_taper, cu::DeviceMemory& d_aterms,
                        cu::DeviceMemory& d_aterm_indices,
                        cu::DeviceMemory& d_metadata,
                        cu::DeviceMemory& d_subgrid);

  void launch_average_beam(int nr_baselines, int nr_antennas, int nr_timesteps,
                           int nr_channels, int nr_aterms, int subgrid_size,
                           cu::DeviceMemory& d_uvw,
                           cu::DeviceMemory& d_baselines,
                           cu::DeviceMemory& d_aterms,
                           cu::DeviceMemory& d_aterm_offsets,
                           cu::DeviceMemory& d_weights,
                           cu::DeviceMemory& d_average_beam);

  void launch_calibrate(
      int nr_subgrids, int grid_size, int subgrid_size, float image_size,
      float w_step, int total_nr_timesteps, int nr_channels, int nr_stations,
      int nr_terms, cu::DeviceMemory& d_uvw, cu::DeviceMemory& d_wavenumbers,
      cu::DeviceMemory& d_visibilities, cu::DeviceMemory& d_weights,
      cu::DeviceMemory& d_aterm, cu::DeviceMemory& d_aterm_derivatives,
      cu::DeviceMemory& d_aterm_indices, cu::DeviceMemory& d_metadata,
      cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_sums1,
      cu::DeviceMemory& d_sums2, cu::DeviceMemory& d_lmnp,
      cu::DeviceMemory& d_hessian, cu::DeviceMemory& d_gradient,
      cu::DeviceMemory& d_residual);

  void launch_grid_fft(cu::DeviceMemory& d_data, int batch, long size,
                       DomainAtoDomainB direction);

  std::unique_ptr<KernelFFT> plan_batched_fft(size_t size, size_t batch);

  void launch_batched_fft(KernelFFT& kernel, cu::DeviceMemory& d_data,
                          size_t batch, DomainAtoDomainB direction);

  void launch_fft_shift(cu::DeviceMemory& d_data, int batch, long size,
                        std::complex<float> scale = {1.0, 1.0});

  void launch_adder(int nr_subgrids, int nr_polarizations, long grid_size,
                    int subgrid_size, cu::DeviceMemory& d_metadata,
                    cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_grid);

  void launch_splitter(int nr_subgrids, int nr_polarizations, long grid_size,
                       int subgrid_size, cu::DeviceMemory& d_metadata,
                       cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_grid);

  void launch_scaler(int nr_subgrids, int nr_polarizations, int subgrid_size,
                     cu::DeviceMemory& d_subgrid);

  void launch_copy_tiles(unsigned int nr_polarizations, unsigned int nr_tiles,
                         unsigned int src_tile_size, unsigned int dst_tile_size,
                         cu::DeviceMemory& d_src_tile_ids,
                         cu::DeviceMemory& d_dst_tile_ids,
                         cu::DeviceMemory& d_src_tiles,
                         cu::DeviceMemory& d_dst_tiles);

  void launch_apply_phasor_to_wtiles(unsigned int nr_polarizations,
                                     unsigned int nr_tiles, float image_size,
                                     float w_step, unsigned int tile_size,
                                     cu::DeviceMemory& d_tiles,
                                     cu::DeviceMemory& d_shift,
                                     cu::DeviceMemory& d_tile_coordinates,
                                     int sign = -1);

  void launch_adder_subgrids_to_wtiles(int nr_subgrids, int nr_polarizations,
                                       long grid_size, int subgrid_size,
                                       int tile_size, int subgrid_offset,
                                       cu::DeviceMemory& d_metadata,
                                       cu::DeviceMemory& d_subgrid,
                                       cu::DeviceMemory& d_tiles,
                                       std::complex<float> scale = {1.0, 1.0});

  void launch_adder_wtiles_to_grid(int nr_polarizations, int nr_tiles,
                                   long grid_size, int tile_size,
                                   int padded_tile_size,
                                   cu::DeviceMemory& d_tile_ids,
                                   cu::DeviceMemory& d_tile_coordinates,
                                   cu::DeviceMemory& d_tiles,
                                   cu::DeviceMemory& d_grid);

  void launch_splitter_subgrids_from_wtiles(
      int nr_subgrids, int nr_polarizations, long grid_size, int subgrid_size,
      int tile_size, int subgrid_offset, cu::DeviceMemory& d_metadata,
      cu::DeviceMemory& d_subgrid, cu::DeviceMemory& d_tiles);

  void launch_splitter_wtiles_from_grid(int nr_polarizations, int nr_tiles,
                                        long grid_size, int tile_size,
                                        int padded_tile_size,
                                        cu::DeviceMemory& d_tile_ids,
                                        cu::DeviceMemory& d_tile_coordinates,
                                        cu::DeviceMemory& d_tiles,
                                        cu::DeviceMemory& d_grid);

  void launch_adder_wtiles_to_patch(
      int nr_polarizations, int nr_tiles, long grid_size, int tile_size,
      int padded_tile_size, int patch_size, idg::Coordinate patch_coordinate,
      cu::DeviceMemory& d_tile_ids, cu::DeviceMemory& d_tile_coordinates,
      cu::DeviceMemory& d_tiles, cu::DeviceMemory& d_patch);

  void launch_splitter_wtiles_from_patch(
      int nr_polarizations, int nr_tiles, long grid_size, int tile_size,
      int padded_tile_size, int patch_size, idg::Coordinate patch_coordinate,
      cu::DeviceMemory& d_tile_ids, cu::DeviceMemory& d_tile_coordinates,
      cu::DeviceMemory& d_tiles, cu::DeviceMemory& d_patch);

  void free_events();

  size_t get_free_memory() const;
  size_t get_total_memory() const;

  void enqueue_report(cu::Stream& stream, int nr_polarizations,
                      int nr_timesteps, int nr_subgrids);

 private:
  // Since no CUDA calls are allowed from a callback, we have to
  // keep track of the cu::Events used in the UpdateData and
  // free them explicitely using the free_events() method.
  cu::Event& get_event();
  std::vector<std::unique_ptr<cu::Event>> events;

  std::unique_ptr<cu::Context> context_;
  std::unique_ptr<cu::Device> device_;
  std::unique_ptr<cu::Stream>
      stream_execute_;  ///< Stream for kernel execution.
  std::unique_ptr<cu::Stream>
      stream_htod_;  ///< Stream for host to device memory copies.
  std::unique_ptr<cu::Stream>
      stream_dtoh_;  ///< Stream for device to host memory copies.

  void compile_gridder(bool enable_avg_aterm);
  void compile_degridder();
  void compile_scaler();
  void compile_adder();
  void compile_splitter();
  void compile_calibrate();
  void compile_fft_shift();
  void compile_average_beam();
  void compile_wtiling();

  std::unique_ptr<KernelFactory<KernelGridder>> factory_gridder_;
  std::unique_ptr<KernelGridder> kernel_gridder_;

  std::unique_ptr<KernelFactory<KernelDegridder>> factory_degridder_;
  std::unique_ptr<KernelDegridder> kernel_degridder_;

  std::unique_ptr<KernelFactory<KernelScaler>> factory_scaler_;
  std::unique_ptr<KernelScaler> kernel_scaler_;

  std::unique_ptr<KernelFactory<KernelAdder>> factory_adder_;
  std::unique_ptr<KernelAdder> kernel_adder_;

  std::unique_ptr<KernelFactory<KernelSplitter>> factory_splitter_;
  std::unique_ptr<KernelSplitter> kernel_splitter_;

  std::unique_ptr<KernelFactory<KernelCalibrateLMNP>> factory_calibrate_lmnp_;
  std::unique_ptr<KernelCalibrateLMNP> kernel_calibrate_lmnp_;

  std::unique_ptr<KernelFactory<KernelCalibrateSums>> factory_calibrate_sums_;
  std::unique_ptr<KernelCalibrateSums> kernel_calibrate_sums_;

  std::unique_ptr<KernelFactory<KernelCalibrateGradient>>
      factory_calibrate_gradient_;
  std::unique_ptr<KernelCalibrateGradient> kernel_calibrate_gradient_;

  std::unique_ptr<KernelFactory<KernelCalibrateHessian>>
      factory_calibrate_hessian_;
  std::unique_ptr<KernelCalibrateHessian> kernel_calibrate_hessian_;

  std::unique_ptr<KernelFactory<KernelFFTShift>> factory_fft_shift_;
  std::unique_ptr<KernelFFTShift> kernel_fft_shift_;

  std::unique_ptr<KernelFactory<KernelAverageBeam>> factory_average_beam_;
  std::unique_ptr<KernelAverageBeam> kernel_average_beam_;

  std::unique_ptr<KernelFactory<KernelWTilingCopy>> factory_wtiling_copy_;
  std::unique_ptr<KernelWTilingCopy> kernel_wtiling_copy_;

  std::unique_ptr<KernelFactory<KernelWTilingPhasor>> factory_wtiling_phasor_;
  std::unique_ptr<KernelWTilingPhasor> kernel_wtiling_phasor_;

  std::unique_ptr<KernelFactory<KernelWTilingSubgridsFromWtiles>>
      factory_wtiling_subgrids_from_wtiles_;
  std::unique_ptr<KernelWTilingSubgridsFromWtiles>
      kernel_wtiling_subgrids_from_wtiles_;

  std::unique_ptr<KernelFactory<KernelWTilingSubgridsToWtiles>>
      factory_wtiling_subgrids_to_wtiles_;
  std::unique_ptr<KernelWTilingSubgridsToWtiles>
      kernel_wtiling_subgrids_to_wtiles_;

  std::unique_ptr<KernelFactory<KernelWTilingWTilesFromPatch>>
      factory_wtiling_wtiles_from_patch_;
  std::unique_ptr<KernelWTilingWTilesFromPatch>
      kernel_wtiling_wtiles_from_patch_;

  std::unique_ptr<KernelFactory<KernelWTilingWtilesToPatch>>
      factory_wtiling_wtiles_to_patch_;
  std::unique_ptr<KernelWTilingWtilesToPatch> kernel_wtiling_wtiles_to_patch_;

  std::unique_ptr<KernelFactory<KernelWTilingWtilesFromGrid>>
      factory_wtiling_wtiles_from_grid_;
  std::unique_ptr<KernelWTilingWtilesFromGrid> kernel_wtiling_wtiles_from_grid_;

  std::unique_ptr<KernelFactory<KernelWTilingWtilesToGrid>>
      factory_wtiling_wtiles_to_grid_;
  std::unique_ptr<KernelWTilingWtilesToGrid> kernel_wtiling_wtiles_to_grid_;

  void start_measurement(void* data);
  void end_measurement(void* data);
};
std::ostream& operator<<(std::ostream& os, InstanceCUDA& d);

}  // end namespace idg::kernel::cuda

#endif
