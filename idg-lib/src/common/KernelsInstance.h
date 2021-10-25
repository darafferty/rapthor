// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_KERNELS_H_
#define IDG_KERNELS_H_

#include <cassert>

#ifndef NDEBUG
#define ASSERT(x) assert(x)
#else
#define ASSERT(x) ((void)(x))
#endif

#include "Report.h"

#include "idg-common.h"

namespace idg {
namespace kernel {

class KernelsInstance {
 public:
  /*
      Misc math routines
  */
  void shift(Array3D<std::complex<float>>& data);

  void scale(Array3D<std::complex<float>>& data,
             std::complex<float> scale) const;

  void tile_backward(const unsigned int nr_polarizations,
                     const unsigned long grid_size,
                     const unsigned int tile_size,
                     const Array5D<std::complex<float>>& grid_src,
                     Grid& grid_dst) const;

  void tile_forward(const unsigned int nr_polarizations,
                    const unsigned long grid_size, const unsigned int tile_size,
                    const Grid& grid_src,
                    Array5D<std::complex<float>>& grid_dst) const;

  void transpose_aterm(
      const unsigned int nr_polarizations,
      const Array4D<Matrix2x2<std::complex<float>>>& aterms_src,
      Array4D<std::complex<float>>& aterms_dst) const;

  /*
      Debug
   */
  void print_memory_info();

  /*
      Performance reporting
  */
 public:
  void set_report(std::shared_ptr<Report> report) { m_report = report; }

 protected:
  std::shared_ptr<Report> m_report;
  std::shared_ptr<powersensor::PowerSensor> m_powersensor;

};  // end class KernelsInstance

}  // namespace kernel
}  // namespace idg

#endif
