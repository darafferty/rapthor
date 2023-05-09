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

  void set_grid(aocommon::xt::Span<std::complex<float>, 4>& grid) override;

  aocommon::xt::Span<std::complex<float>, 4>& get_final_grid() override;

  void init_cache(int subgrid_size, float cell_size, float w_step,
                  const std::array<float, 2>& shift) override;

  std::unique_ptr<Plan> make_plan(
      const int kernel_size, const aocommon::xt::Span<float, 1>& frequencies,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      Plan::Options options) override;

 private:
  void run_imaging(
      const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
      aocommon::xt::Span<std::complex<float>, 4>& visibilities,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      aocommon::xt::Span<std::complex<float>, 4>& grid,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      const aocommon::xt::Span<float, 2>& taper, ImagingMode mode);

  /*
   * Gridding
   */
  void do_gridding(
      const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
      const aocommon::xt::Span<std::complex<float>, 4>& visibilities,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      const aocommon::xt::Span<float, 2>& taper) override;

  /*
   * Degridding
   */
  void do_degridding(
      const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
      aocommon::xt::Span<std::complex<float>, 4>& visibilities,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      const aocommon::xt::Span<float, 2>& taper) override;

  /*
   * FFT
   */
  void do_transform(DomainAtoDomainB direction) override;

  /*
   * Calibration
   */
  void do_calibrate_init(
      std::vector<std::vector<std::unique_ptr<Plan>>>&& plans,
      const aocommon::xt::Span<float, 2>& frequencies,
      Tensor<std::complex<float>, 6>&& visibilities, Tensor<float, 6>&& weights,
      Tensor<UVW<float>, 3>&& uvw,
      Tensor<std::pair<unsigned int, unsigned int>, 2>&& baselines,
      const aocommon::xt::Span<float, 2>& taper) override;

  void do_calibrate_update(
      const int antenna_nr,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 5>& aterms,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 5>&
          aterm_derivatives,
      aocommon::xt::Span<double, 4>& hessian,
      aocommon::xt::Span<double, 3>& gradient,
      aocommon::xt::Span<double, 1>& residual) override;

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
    std::vector<std::vector<Tensor<std::complex<float>, 4>>> subgrids;
    std::vector<Tensor<float, 1>> wavenumbers;
    Tensor<std::complex<float>, 6> visibilities;
    Tensor<float, 6> weights;
    Tensor<UVW<float>, 3> uvw;
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
    std::vector<std::unique_ptr<cu::DeviceMemory>> d_aterm_indices;
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
