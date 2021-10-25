// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_HYBRID_GENERIC_OPTIMIZED_H_
#define IDG_HYBRID_GENERIC_OPTIMIZED_H_

#include <thread>

#include "idg-cpu.h"
#include "CUDA/common/CUDA.h"

namespace idg {
namespace proxy {
namespace hybrid {

class GenericOptimized : public cuda::CUDA {
 public:
  GenericOptimized();
  ~GenericOptimized();

  virtual bool do_supports_wstacking() {
    return cpuProxy->do_supports_wstacking();
  }

  virtual bool do_supports_wtiling() override {
    return !m_disable_wtiling &&
           (!m_disable_wtiling_gpu || cpuProxy->supports_wtiling());
  }

  void set_disable_wtiling(bool v) override {
    m_disable_wtiling = v;
    cpuProxy->set_disable_wtiling(v);
  }

  void set_disable_wtiling_gpu(bool v) { m_disable_wtiling_gpu = v; }

  void set_grid(std::shared_ptr<Grid> grid) override;

  std::shared_ptr<Grid> get_final_grid() override;

  void init_cache(int subgrid_size, float cell_size, float w_step,
                  const Array1D<float>& shift) override;

  std::unique_ptr<Plan> make_plan(
      const int kernel_size, const Array1D<float>& frequencies,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array1D<unsigned int>& aterms_offsets,
      Plan::Options options) override;

 private:
  /*
   * Gridding
   */
  void do_gridding(
      const Plan& plan, const Array1D<float>& frequencies,
      const Array4D<std::complex<float>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  void run_gridding(
      const Plan& plan, const Array1D<float>& frequencies,
      const Array4D<std::complex<float>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal);

  /*
   * Degridding
   */
  void do_degridding(
      const Plan& plan, const Array1D<float>& frequencies,
      Array4D<std::complex<float>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  void run_degridding(
      const Plan& plan, const Array1D<float>& frequencies,
      Array4D<std::complex<float>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal);

  /*
   * FFT
   */
  void do_transform(DomainAtoDomainB direction) override;

  /*
   * Calibration
   */
  void do_calibrate_init(
      std::vector<std::unique_ptr<Plan>>&& plans,
      const Array1D<float>& frequencies,
      Array5D<std::complex<float>>&& visibilities, Array5D<float>&& weights,
      Array3D<UVW<float>>&& uvw,
      Array2D<std::pair<unsigned int, unsigned int>>&& baselines,
      const Array2D<float>& spheroidal) override;

  void do_calibrate_update(
      const int station_nr,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
      Array3D<double>& hessian, Array2D<double>& gradient,
      double& residual) override;

  void do_calibrate_finish() override;

  /*
   * W-Tiling
   */
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

 protected:
  idg::proxy::cpu::CPU* cpuProxy;

  /*
   * W-Tiling state
   */
  bool m_disable_wtiling_gpu = false;
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
   * Calibration state
   */
  struct {
    std::vector<std::unique_ptr<Plan>> plans;
    unsigned int nr_baselines;
    unsigned int nr_timesteps;
    unsigned int nr_channels;
    Array3D<UVW<float>> uvw;
  } m_calibrate_state;

  // Note:
  //     kernel_calibrate internally assumes max_nr_terms = 8
  //     and will process larger values of nr_terms in batches
  const unsigned int m_calibrate_max_nr_terms = 8;

};  // class GenericOptimized

}  // namespace hybrid
}  // namespace proxy
}  // namespace idg

#endif
