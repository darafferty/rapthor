// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

struct Plan* Plan_create(
    const int kernel_size, const int subgrid_size, const int grid_size,
    const float cell_size, float* frequencies,
    const unsigned int frequencies_nr_channels, float* uvw,
    const unsigned int uvw_nr_baselines, const unsigned int uvw_nr_timesteps,
    const unsigned int uvw_nr_coordinates, unsigned int* baselines,
    const unsigned int baselines_nr_baselines, const unsigned int baselines_two,
    unsigned int* aterms_offsets,
    const unsigned int aterms_offsets_nr_timeslots_plus_one);

int Plan_get_nr_subgrids(struct Plan* plan);
void Plan_copy_metadata(struct Plan* plan, void* ptr);
void Plan_destroy(struct Plan* plan);

