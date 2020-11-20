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

class Unified : public Generic {
 public:
  // Constructor
  Unified(ProxyInfo info = default_info());

  // Destructor
  ~Unified();

  virtual void do_gridding(
      const Plan& plan,
      const float w_step,  // in lambda
      const Array1D<float>& shift, const float cell_size,
      const unsigned int kernel_size,  // full width in pixels
      const unsigned int subgrid_size, const Array1D<float>& frequencies,
      const Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  virtual void do_degridding(
      const Plan& plan,
      const float w_step,  // in lambda
      const Array1D<float>& shift, const float cell_size,
      const unsigned int kernel_size,  // full width in pixels
      const unsigned int subgrid_size, const Array1D<float>& frequencies,
      Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  virtual void do_transform(DomainAtoDomainB direction,
                            Array3D<std::complex<float>>& grid) override;

  virtual void set_grid(std::shared_ptr<Grid> grid) override;

  virtual std::shared_ptr<Grid> get_grid() override;

 private:
  // The m_grid member defined in Proxy
  // may not reside in Unified Memory.
  // This m_grid_tiled is initialized as a copy
  // of m_grid and (optionally) tiled to match the
  // data access pattern in the unified_adder
  // and unified_splitter kernels.
  std::unique_ptr<Grid> m_grid_tiled = nullptr;

};  // class Unified

}  // namespace cuda
}  // namespace proxy
}  // namespace idg

#endif
