// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_CUDA_H_
#define IDG_CUDA_H_

#include <vector>
#include <complex>

#include "idg-common.h"
namespace cu {
class DeviceMemory;
class HostMemory;
class UnifiedMemory;
class Context;
};  // namespace cu

namespace cufft {
class C2C_2D;
};  // namespace cufft

namespace idg {
namespace kernel {
namespace cuda {
class InstanceCUDA;
}
}  // namespace kernel

namespace proxy {
namespace cuda {
class CUDA : public Proxy {
 public:
  CUDA(ProxyInfo info);

  ~CUDA();

 public:
  std::unique_ptr<auxiliary::Memory> allocate_memory(size_t bytes) override;

  void print_compiler_flags();

  void print_devices();

  unsigned int get_num_devices() const;
  idg::kernel::cuda::InstanceCUDA& get_device(unsigned int i) const;

  static ProxyInfo default_info();

  /*
   * Beam
   */
  virtual void do_compute_avg_beam(
      const unsigned int nr_antennas, const unsigned int nr_channels,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array4D<float>& weights,
      idg::Array4D<std::complex<float>>& average_beam) override;

 protected:
  void init_devices();

  std::unique_ptr<powersensor::PowerSensor> hostPowerSensor;

  /*
   * Options used internally by the CUDA proxies
   */
  // Fraction of device memory reserved
  // for e.g. cuFFT. This memory is not taken
  // into account when computing  in compute_jobsize.
  float m_fraction_reserved = 0.15;

  // Use Unified Memory to store the grid, instead of having
  // a copy on the grid on the device.
  bool m_use_unified_memory = false;

 public:
  void set_fraction_reserved(float f) { m_fraction_reserved = f; }
  void enable_unified_memory() { m_use_unified_memory = true; }

 protected:
  /*
   * Gridding/degridding
   */
  enum ImagingMode { mode_gridding, mode_degridding };

  struct JobData {
    unsigned first_bl;
    unsigned current_time_offset;
    unsigned current_nr_baselines;
    unsigned current_nr_subgrids;
    unsigned current_nr_timesteps;
    const idg::Metadata* metadata_ptr;
    const idg::UVW<float>* uvw_ptr;
    std::complex<float>* visibilities_ptr;
  };

  int initialize_jobs(const int nr_baselines, const int nr_timesteps,
                      const int nr_channels, const int subgrid_size,
                      const size_t bytes_free, const Plan& plan,
                      const Array4D<std::complex<float>>& visibilities,
                      const Array2D<UVW<float>>& uvw,
                      std::vector<JobData>& jobs) const;

  /*
   * W-Tiling
   */
  void init_buffers_wtiling(unsigned int subgrid_size);
  void free_buffers_wtiling();

  unsigned int plan_tile_fft(unsigned int nr_polarizations,
                             unsigned int nr_tiles_batch,
                             const unsigned int w_padded_tile_size,
                             const cu::Context& context,
                             const size_t free_memory,
                             std::unique_ptr<cufft::C2C_2D>& fft) const;

  size_t bytes_required_wtiling(const WTileUpdateSet& wtile_set,
                                const int nr_polarizations,
                                const int subgrid_size, const float image_size,
                                const float w_step,
                                const idg::Array1D<float>& shift,
                                const size_t bytes_free) const;

  void run_wtiles_to_grid(unsigned int subgrid_size, float image_size,
                          float w_step, const Array1D<float>& shift,
                          WTileUpdateInfo& wtile_flush_info);

  void run_subgrids_to_wtiles(unsigned int nr_polarizations,
                              unsigned int subgrid_offset,
                              unsigned int nr_subgrids,
                              unsigned int subgrid_size, float image_size,
                              float w_step, const Array1D<float>& shift,
                              WTileUpdateSet& wtile_flush_set,
                              cu::DeviceMemory& d_subgrids,
                              cu::DeviceMemory& d_metadata);

  void run_wtiles_from_grid(unsigned int subgrid_size, float image_size,
                            float w_step, const Array1D<float>& shift,
                            WTileUpdateInfo& wtile_initialize_info);

  void run_subgrids_from_wtiles(unsigned int nr_polarizations,
                                unsigned int subgrid_offset,
                                unsigned int nr_subgrids,
                                unsigned int subgrid_size, float image_size,
                                float w_step, const Array1D<float>& shift,
                                WTileUpdateSet& wtile_initialize_set,
                                cu::DeviceMemory& d_subgrids,
                                cu::DeviceMemory& d_metadata);

  void flush_wtiles();

  WTiles m_wtiles;
  unsigned int m_nr_tiles = 0;  // configured in init_cache
  const unsigned int m_tile_size = 128;
  const unsigned int m_patch_size = 512;
  const unsigned int m_nr_patches_batch = 3;

  struct {
    std::unique_ptr<cu::DeviceMemory> d_tiles;
    std::unique_ptr<cu::DeviceMemory> d_padded_tiles;
    std::unique_ptr<cu::HostMemory> h_tiles;
    std::vector<std::unique_ptr<cu::DeviceMemory>> d_patches;
  } m_buffers_wtiling;

  /*
   * Unified Memory
   */
  cu::UnifiedMemory& get_unified_grid() { return *u_grid_; }
  cu::UnifiedMemory& allocate_unified_grid(const cu::Context& context,
                                           size_t size);
  void free_unified_grid();

 private:
  ProxyInfo& mInfo;
  std::vector<std::unique_ptr<kernel::cuda::InstanceCUDA>> devices;

  /**
   * Copy of the grid in CUDA Unified Memory used by the Generic and Unified
   * proxies.
   */
  std::unique_ptr<cu::UnifiedMemory> u_grid_;
};
}  // namespace cuda
}  // end namespace proxy
}  // end namespace idg

#endif
