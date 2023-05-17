// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_CUDA_GENERIC_H_
#define IDG_CUDA_GENERIC_H_

#include "idg-common.h"
#include "CUDA/common/CUDA.h"

namespace idg {
namespace proxy {
namespace cuda {
/**
 * @brief Generic CUDA Proxy
 *
 */
class Generic : public CUDA {
 public:
  // Constructor
  Generic(ProxyInfo info = default_info());

  // Destructor
  ~Generic();

 private:
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

  void do_transform(DomainAtoDomainB direction) override;

  void run_imaging(
      const Plan& plan, const aocommon::xt::Span<float, 1>& frequencies,
      const aocommon::xt::Span<std::complex<float>, 4>& visibilities,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      aocommon::xt::Span<std::complex<float>, 4>& grid,
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      const aocommon::xt::Span<float, 2>& taper, ImagingMode mode);

 public:
  bool do_supports_wtiling() override { return true; }

  void set_grid(aocommon::xt::Span<std::complex<float>, 4>& grid) override;

  aocommon::xt::Span<std::complex<float>, 4>& get_final_grid() override;

  virtual std::unique_ptr<Plan> make_plan(
      const int kernel_size, const aocommon::xt::Span<float, 1>& frequencies,
      const aocommon::xt::Span<UVW<float>, 2>& uvw,
      const aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1>&
          baselines,
      const aocommon::xt::Span<unsigned int, 1>& aterm_offsets,
      Plan::Options options) override;

  void init_cache(int subgrid_size, float cell_size, float w_step,
                  const std::array<float, 2>& shift) override;

 private:
  std::unique_ptr<cu::DeviceMemory> d_grid_;
};  // class Generic

}  // namespace cuda
}  // namespace proxy
}  // namespace idg
#endif
