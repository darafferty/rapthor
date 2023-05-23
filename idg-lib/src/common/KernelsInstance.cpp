// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "KernelsInstance.h"

#include <cassert>
#include <csignal>

#include <xtensor/xview.hpp>

namespace idg::kernel {

void KernelsInstance::fftshift_grid(
    aocommon::xt::Span<std::complex<float>, 3>& grid) {
  const size_t nr_polarizations = grid.shape(0);
  const size_t height = grid.shape(1);
  const size_t width = grid.shape(2);
  assert(height == width);

  pmt::State states[2];
  states[0] = power_meter_->Read();

  std::complex<float> tmp13, tmp24;

  // Dimensions
  const size_t n = height;
  const size_t n2 = n / 2;

  // Interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
  for (size_t pol = 0; pol < nr_polarizations; pol++) {
#pragma omp parallel for private(tmp13, tmp24) collapse(2)
    for (size_t i = 0; i < n2; i++) {
      for (size_t k = 0; k < n2; k++) {
        tmp13 = grid(pol, i, k);
        grid(pol, i, k) = grid(pol, i + n2, k + n2);
        grid(pol, i + n2, k + n2) = tmp13;

        tmp24 = grid(pol, i + n2, k);
        grid(pol, i + n2, k) = grid(pol, i, k + n2);
        grid(pol, i, k + n2) = tmp24;
      }
    }
  }

  states[1] = power_meter_->Read();
  report_->update<Report::ID::fft_shift>(states[0], states[1]);
}

void KernelsInstance::tile_grid(
    aocommon::xt::Span<std::complex<float>, 4>& grid_untiled,
    aocommon::xt::Span<std::complex<float>, 5>& grid_tiled,
    bool forward) const {
  const size_t nr_w_layers = grid_untiled.shape(0);
  const size_t untiled_nr_polarizations = grid_untiled.shape(1);
  const size_t grid_height = grid_untiled.shape(2);
  const size_t grid_width = grid_untiled.shape(3);
  const size_t grid_size = grid_height;

  const size_t nr_tiles_y = grid_tiled.shape(0);
  const size_t nr_tiles_x = grid_tiled.shape(1);
  const size_t nr_tiles_1d = nr_tiles_y;
  const size_t tiled_nr_polarizations = grid_tiled.shape(2);
  const size_t tile_height = grid_tiled.shape(3);
  const size_t tile_width = grid_tiled.shape(4);
  const size_t tile_size = tile_height;

  assert(nr_w_layers == 1);
  assert(grid_height == grid_width);
  assert(nr_tiles_y == nr_tiles_x);
  assert(untiled_nr_polarizations == tiled_nr_polarizations);
  assert(tile_height == tile_width);
  assert(nr_tiles_1d * tile_size == grid_size);

#pragma omp parallel for
  for (size_t pixel = 0; pixel < grid_size * grid_size; pixel++) {
    const size_t grid_y = pixel / grid_size;
    const size_t grid_x = pixel % grid_size;
    const size_t tile_id_y = grid_y / tile_size;
    const size_t tile_id_x = grid_x / tile_size;
    const size_t tile_y = grid_y % tile_size;
    const size_t tile_x = grid_x % tile_size;
    auto tiled_view =
        xt::view(grid_tiled, tile_id_y, tile_id_x, xt::all(), tile_y, tile_x);
    auto untiled_view = xt::view(grid_untiled, 0, xt::all(), grid_y, grid_x);
    if (forward) {
      tiled_view = untiled_view;
    } else {
      untiled_view = tiled_view;
    }
  }
}

void KernelsInstance::transpose_aterm(
    const unsigned int nr_polarizations,
    const aocommon::xt::Span<Matrix2x2<std::complex<float>>, 4>& aterms_src,
    aocommon::xt::Span<std::complex<float>, 4>& aterms_dst) const {
  assert(nr_polarizations * aterms_src.size() == aterms_dst.size());
  const size_t nr_stations = aterms_src.shape(0);
  const size_t nr_timeslots = aterms_src.shape(1);
  const size_t subgrid_size = aterms_src.shape(2);
  assert(subgrid_size == aterms_src.shape(3));
  assert(nr_polarizations == aterms_dst.shape(1));

#pragma omp parallel for
  for (size_t pixel = 0; pixel < subgrid_size * subgrid_size; pixel++) {
    for (size_t station = 0; station < nr_stations; station++) {
      for (size_t timeslot = 0; timeslot < nr_timeslots; timeslot++) {
        const size_t y = pixel / subgrid_size;
        const size_t x = pixel % subgrid_size;
        const size_t term_nr = station * nr_timeslots + timeslot;

        Matrix2x2<std::complex<float>> term =
            aterms_src(station, timeslot, y, x);
        aterms_dst(term_nr, 0, y, x) = term.xx;
        aterms_dst(term_nr, 1, y, x) = term.xy;
        aterms_dst(term_nr, 2, y, x) = term.yx;
        aterms_dst(term_nr, 3, y, x) = term.yy;
      }
    }
  }
}

void KernelsInstance::print_memory_info() {
  auto memory_total = auxiliary::get_total_memory() / float(1024);  // GBytes
  auto memory_used = auxiliary::get_used_memory() / float(1024);    // GBytes
  auto memory_free = memory_total - memory_used;
  std::clog << "Host memory -> " << std::fixed << std::setprecision(1);
  std::clog << "total: " << memory_total << " Gb, ";
  std::clog << "used: " << memory_used << " Gb, ";
  std::clog << "free: " << memory_free << " Gb" << std::endl;
}

}  // namespace idg::kernel
