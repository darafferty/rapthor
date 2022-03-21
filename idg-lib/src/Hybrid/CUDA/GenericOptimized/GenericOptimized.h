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

/**
 * @brief Hybrid Proxy, combines functionality from CPU Optimized and CUDA
 * Generic proxies
 *
 */
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
  void run_imaging(
      const Plan& plan, const Array1D<float>& frequencies,
      Array4D<std::complex<float>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal, ImagingMode mode);

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
      std::vector<std::vector<std::unique_ptr<Plan>>>&& plans,
      const Array2D<float>& frequencies,
      Array6D<std::complex<float>>&& visibilities, Array6D<float>&& weights,
      Array3D<UVW<float>>&& uvw,
      Array2D<std::pair<unsigned int, unsigned int>>&& baselines,
      const Array2D<float>& spheroidal) override;

  void do_calibrate_update(
      const int station_nr,
      const Array5D<Matrix2x2<std::complex<float>>>& aterms,
      const Array5D<Matrix2x2<std::complex<float>>>& derivative_aterms,
      Array4D<double>& hessian, Array3D<double>& gradient,
      Array1D<double>& residual) override;

  void do_calibrate_finish() override;

 protected:
  idg::proxy::cpu::CPU* cpuProxy;

  /*
   * W-Tiling state
   */
  bool m_disable_wtiling_gpu = false;

  /*
   * Calibration state
   */
  struct {
    std::vector<std::vector<std::unique_ptr<Plan>>> plans;
    std::vector<std::vector<Array4D<std::complex<float>>>> subgrids;
    std::vector<Array1D<float>> wavenumbers;
    Array6D<std::complex<float>> visibilities;
    Array6D<float> weights;
    Array3D<UVW<float>> uvw;
    unsigned int nr_baselines;
    unsigned int nr_timesteps;
    unsigned int nr_channels_per_block;
    unsigned int nr_channel_blocks;
    std::unique_ptr<cu::DeviceMemory> d_wavenumbers;
    std::unique_ptr<cu::DeviceMemory> d_lmnp;
    std::unique_ptr<cu::DeviceMemory> d_sums_x;
    std::unique_ptr<cu::DeviceMemory> d_sums_y;
    std::vector<std::unique_ptr<cu::DeviceMemory>> d_metadata;
    std::vector<std::unique_ptr<cu::DeviceMemory>> d_subgrids;
    std::vector<std::unique_ptr<cu::DeviceMemory>> d_visibilities;
    std::vector<std::unique_ptr<cu::DeviceMemory>> d_weights;
    std::vector<std::unique_ptr<cu::DeviceMemory>> d_uvw;
    std::vector<std::unique_ptr<cu::DeviceMemory>> d_aterms_indices;
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
