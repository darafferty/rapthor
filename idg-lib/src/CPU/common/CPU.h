// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_CPU_H_
#define IDG_CPU_H_

#include "idg-common.h"

#include "InstanceCPU.h"

namespace idg {
namespace proxy {
namespace cpu {

class CPU : public Proxy {
 public:
  // Constructor
  CPU();

  // Disallow assignment and pass-by-value
  CPU& operator=(const CPU& rhs) = delete;
  CPU(const CPU& v) = delete;

  // Destructor
  virtual ~CPU();

  std::unique_ptr<auxiliary::Memory> allocate_memory(size_t bytes) override;

  virtual bool do_supports_wstacking() override {
    return m_kernels->do_supports_wstacking();
  }

  virtual bool do_supports_wtiling() override {
    return m_kernels->do_supports_wtiling();
  }

  std::shared_ptr<kernel::cpu::InstanceCPU> get_kernels() { return m_kernels; }

  std::unique_ptr<Plan> make_plan(
      const int kernel_size, const aocommon::xt::Span<float, 1>& frequencies,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      Plan::Options options) override;

  void init_cache(int subgrid_size, float cell_size, float w_step,
                  const std::array<float, 2>& shift) override;

  aocommon::xt::Span<std::complex<float>, 4>& get_final_grid() override;

 private:
  unsigned int compute_jobsize(const Plan& plan,
                               const unsigned int nr_timesteps,
                               const unsigned int nr_channels,
                               const unsigned int nr_correlations,
                               const unsigned int nr_polarizations,
                               const unsigned int subgrid_size);

  // Routines
  void do_gridding(
      const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
      const aocommon::xt::Span<std::complex<float>, 4>& visibilities,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      const aocommon::xt::Span<float, 2>& taper) override;

  void do_degridding(
      const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
      aocommon::xt::Span<std::complex<float>, 4>& visibilities,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      const aocommon::xt::Span<float, 2>& taper) override;

  void do_calibrate_init(
      std::vector<std::vector<std::unique_ptr<Plan>>>&& plans,
      const aocommon::xt::Span<float, 2>& frequencies,
      Tensor<std::complex<float>, 6>&& visibilities, Tensor<float, 6>&& weights,
      Tensor<UVW<float>, 3>&& uvw,
      Tensor<std::pair<unsigned int, unsigned int>, 2>&& baselines,
      const aocommon::xt::Span<float, 2>& taper) override;

  virtual void do_calibrate_update(
      const int antenna_nr,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 5>& aterms,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 5>&
          aterm_derivatives,
      aocommon::xt::Span<double, 4>& hessian,
      aocommon::xt::Span<double, 3>& gradient,
      aocommon::xt::Span<double, 1>& residual) override;

  void do_calibrate_finish() override;

  void do_transform(DomainAtoDomainB direction) override;

  void do_compute_avg_beam(
      const unsigned int nr_antennas, const unsigned int nr_channels,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      const aocommon::xt::Span<float, 4>& weights,
      aocommon::xt::Span<std::complex<float>, 4>& average_beam) override;

 protected:
  void init_wtiles(int grid_size, int subgrid_size, float image_size,
                   float w_step);

  std::shared_ptr<kernel::cpu::InstanceCPU> m_kernels;
  std::unique_ptr<pmt::Pmt> power_meter_;

  /*
   * Options used internally by the CPU proxy
   */
  // Maximum fraction of available memory used to allocate subgrids
  // this value impacts the jobsize that will be used and hence the
  // amount of memory additionaly allocated (if any) in various kernels.
  float m_fraction_memory_subgrids = 0.10;

  // Maximum size of the subgrids buffer allocated in do_gridding
  // and do_degridding. A value of about 10x the size of the L3 cache
  // seems to provide a good balance between the number of kernel calls
  // and the time needed to allocate memory, while it is large enough
  // to provide sufficient scalability.
  size_t m_max_bytes_subgrids = 512 * 1024 * 1024;  // 512 Mb

  WTiles m_wtiles;

  struct {
    std::vector<std::vector<std::unique_ptr<Plan>>> plans;
    size_t nr_baselines;
    size_t nr_timesteps;
    size_t nr_channels;
    Tensor<float, 1> wavenumbers;
    Tensor<std::complex<float>, 6> visibilities;  // ANTxANTxTIMExCHANxCOR
    Tensor<float, 6> weights;                     // ANTxANTxTIMExCHANxCOR
    Tensor<UVW<float>, 3> uvw;
    Tensor<std::pair<unsigned int, unsigned int>, 2> baselines;
    std::vector<Tensor<std::complex<float>, 4>> subgrids;
    std::vector<Tensor<std::complex<float>, 4>> phasors;
    std::vector<int> max_nr_timesteps;
  } m_calibrate_state;

};  // end class CPU

}  // end namespace cpu
}  // end namespace proxy
}  // end namespace idg
#endif
