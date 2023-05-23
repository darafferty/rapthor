// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_KERNELS_INSTANCE_H_
#define IDG_KERNELS_INSTANCE_H_

#include "Report.h"

#include "idg-common.h"

namespace idg::kernel {

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
      const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms_src,
      aocommon::xt::Span<std::complex<float>, 4>& aterms_dst) const;

  /*
      Debug
   */
  void print_memory_info();

  /*
      Performance reporting
  */
 public:
  void set_report(std::shared_ptr<Report> report) { report_ = report; }

 protected:
  std::shared_ptr<Report> report_;
  std::unique_ptr<pmt::Pmt> power_meter_;

};  // end class KernelsInstance

}  // namespace idg::kernel

#endif
