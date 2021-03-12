// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_OPENCL_GENERIC_H_
#define IDG_OPENCL_GENERIC_H_

#include "idg-opencl.h"

namespace powersensor {
class PowerSensor;
}

namespace idg {
namespace proxy {
namespace opencl {
class Generic : public OpenCL {
 public:
  // Constructor
  Generic();

  // Destructor
  ~Generic();

 private:
  void do_gridding(
      const Plan& plan, const Array1D<float>& frequencies,
      const Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  void do_degridding(
      const Plan& plan, const Array1D<float>& frequencies,
      Array3D<Visibility<std::complex<float>>>& visibilities,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms,
      const Array1D<unsigned int>& aterms_offsets,
      const Array2D<float>& spheroidal) override;

  void do_transform(DomainAtoDomainB direction) override;

  powersensor::PowerSensor* hostPowerSensor;

};  // class Generic

}  // namespace opencl
}  // namespace proxy
}  // namespace idg
#endif
