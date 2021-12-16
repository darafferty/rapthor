// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_CUDA_UNIFIED_H_
#define IDG_CUDA_UNIFIED_H_

#include "CUDA/Generic/Generic.h"

namespace powersensor {
class PowerSensor;
}

namespace cu {
class UnifiedMemory;
}

namespace idg {
namespace proxy {
namespace cuda {

/**
 * @brief CUDA Proxy using Unified Memory
 *
 */
class Unified : public Generic {
 public:
  // Constructor
  Unified(ProxyInfo info = default_info());

  // Destructor
  ~Unified();

  void do_gridding(
      const Plan& plan, const Array1D<float>& frequencies,
      const Array4D<std::complex<float>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  void do_degridding(
      const Plan& plan, const Array1D<float>& frequencies,
      Array4D<std::complex<float>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  void do_transform(DomainAtoDomainB direction) override;

  void set_grid(std::shared_ptr<Grid> grid) override;

  std::shared_ptr<Grid> get_final_grid() override;

  bool do_supports_wtiling() override { return false; }

 private:
  // The m_grid member defined in Proxy
  // may not reside in Unified Memory.
  // This m_grid_tiled is initialized as a copy
  // of m_grid and (optionally) tiled to match the
  // data access pattern in the unified_adder
  // and unified_splitter kernels.
  std::unique_ptr<Array5D<std::complex<float>>> m_grid_tiled = nullptr;

};  // class Unified

}  // namespace cuda
}  // namespace proxy
}  // namespace idg

#endif
