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

void Plan_copy_aterms_indices(struct idg::Plan* plan, void* ptr) {
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
                            unsigned int* aterms_offsets,
                            int aterms_offsets_nr_timeslots) {
  idg::Array1D<float> shift_array(3);
  shift_array(0) = 0;
  shift_array(1) = 0;
  shift_array(2) = 0;
  idg::Array1D<float> frequencies_array(frequencies, frequencies_nr_channels);
  assert(uvw_nr_coordinates == 3);
  idg::Array2D<idg::UVW<float>> uvw_array(uvw, uvw_nr_baselines,
                                          uvw_nr_timesteps);
  idg::Array1D<std::pair<unsigned int, unsigned int>> baselines_array(
      baselines, baselines_nr_baselines);
  idg::Array1D<unsigned int> aterms_offsets_array(aterms_offsets,
                                                  aterms_offsets_nr_timeslots);

  return new idg::Plan(kernel_size, subgrid_size, grid_size, cell_size,
                       shift_array, frequencies_array, uvw_array,
                       baselines_array, aterms_offsets_array);
}

}  // extern "C"
