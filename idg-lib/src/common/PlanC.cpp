// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Plan.h"

extern "C" {

int Plan_get_nr_subgrids(struct idg::Plan* plan) {
  return plan->get_nr_subgrids();
}

void Plan_copy_metadata(struct idg::Plan* plan, void* ptr) {
  plan->copy_metadata(ptr);
}

void Plan_copy_aterm_indices(struct idg::Plan* plan, void* ptr) {
  plan->copy_aterm_indices(ptr);
}

void Plan_destroy(struct idg::Plan* plan) { delete plan; }

struct idg::Plan* Plan_init(int kernel_size, int subgrid_size, int grid_size,
                            float cell_size, float* frequencies,
                            int frequencies_nr_channels, idg::UVW<float>* uvw,
                            int uvw_nr_baselines, int uvw_nr_timesteps,
                            int uvw_nr_coordinates,
                            std::pair<unsigned int, unsigned int>* baselines,
                            int baselines_nr_baselines, int baselines_two,
                            unsigned int* aterm_offsets,
                            int aterm_offsets_nr_timeslots) {
  std::array<float, 2> shift{0.0f, 0.0f};
  const std::array<size_t, 1> frequencies_shape{
      static_cast<size_t>(frequencies_nr_channels)};
  aocommon::xt::Span<float, 1> frequencies_span =
      aocommon::xt::CreateSpan(frequencies, frequencies_shape);
  assert(uvw_nr_coordinates == 3);
  const std::array<size_t, 2> uvw_shape{static_cast<size_t>(uvw_nr_baselines),
                                        static_cast<size_t>(uvw_nr_timesteps)};
  aocommon::xt::Span<idg::UVW<float>, 2> uvw_span =
      aocommon::xt::CreateSpan(uvw, uvw_shape);
  const std::array<size_t, 1> baselines_shape{
      static_cast<size_t>(baselines_nr_baselines)};
  aocommon::xt::Span<std::pair<unsigned int, unsigned int>, 1> baselines_span =
      aocommon::xt::CreateSpan(baselines, baselines_shape);
  const std::array<size_t, 1> aterm_offsets_shape{
      static_cast<size_t>(aterm_offsets_nr_timeslots)};
  aocommon::xt::Span<unsigned int, 1> aterm_offsets_span =
      aocommon::xt::CreateSpan(aterm_offsets, aterm_offsets_shape);

  return new idg::Plan(kernel_size, subgrid_size, grid_size, cell_size, shift,
                       frequencies_span, uvw_span, baselines_span,
                       aterm_offsets_span);
}

}  // extern "C"
