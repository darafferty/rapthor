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
  void fftshift_grid(aocommon::xt::Span<std::complex<float>, 3>& grid);

  /// Convert from a untiled grid to a tiled grid (or backwards when !forward)
  /// The grids are assumed to have the following dimensions:
  ///  - grid_untiled: nr_w_layers * nr_polarizations * grid_size * grid_size
  ///  - grid_tiled: nr_tiles_1d * nr_tiles_1d * nr_polarizations * tile_size *
  ///  tile_size
  /// with nr_w_layers == 1 and grid_size == (nr_tiles_1d * tile_size).
  void tile_grid(aocommon::xt::Span<std::complex<float>, 4>& grid_untiled,
                 aocommon::xt::Span<std::complex<float>, 5>& grid_tiled,
                 bool forward = true) const;

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
