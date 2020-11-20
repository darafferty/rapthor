// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_CUDA_GENERIC_H_
#define IDG_CUDA_GENERIC_H_

#include "idg-common.h"
#include "CUDA/common/CUDA.h"

namespace powersensor {
class PowerSensor;
}

namespace idg {
namespace proxy {
namespace cuda {
class Generic : public CUDA {
  friend class Unified;

 public:
  // Constructor
  Generic(ProxyInfo info = default_info());

  // Destructor
  ~Generic();

 private:
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

  powersensor::PowerSensor* hostPowerSensor;

  void run_gridding(
      const Plan& plan, const float w_step, const Array1D<float>& shift,
      const float cell_size, const unsigned int kernel_size,
      const unsigned int subgrid_size, const Array1D<float>& frequencies,
      const Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal);

  void run_degridding(
      const Plan& plan, const float w_step, const Array1D<float>& shift,
      const float cell_size, const unsigned int kernel_size,
      const unsigned int subgrid_size, const Array1D<float>& frequencies,
      Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Grid& grid, const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal);

 public:
  virtual void set_grid(std::shared_ptr<Grid> grid) override;

  virtual std::shared_ptr<Grid> get_grid() override;

};  // class Generic

}  // namespace cuda
}  // namespace proxy
}  // namespace idg
#endif
