// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

extern "C" {

idg::Data* DATA_init(const char* layout_file) {
  return new idg::Data(layout_file);
}

float DATA_compute_image_size(idg::Data* data, unsigned grid_size) {
  return data->compute_image_size(grid_size);
}

float DATA_compute_max_uv(idg::Data* data, unsigned long grid_size) {
  return data->compute_max_uv(grid_size);
}

unsigned int DATA_compute_grid_size(idg::Data* data) {
  return data->compute_grid_size();
}

void DATA_limit_max_baseline_length(idg::Data* data, float max_uv) {
  data->limit_max_baseline_length(max_uv);
}

void DATA_limit_nr_baselines(idg::Data* data, unsigned int n) {
  data->limit_nr_baselines(n);
}

void DATA_limit_nr_stations(idg::Data* data, unsigned int n) {
  data->limit_nr_stations(n);
}

float DATA_get_nr_stations(idg::Data* data) { return data->get_nr_stations(); }

float DATA_get_nr_baselines(idg::Data* data) {
  return data->get_nr_baselines();
}

void DATA_get_frequencies(idg::Data* data, void* ptr, unsigned int nr_channels,
                          float image_size, unsigned int channel_offset) {
  idg::Array1D<float> frequencies((float*)ptr, nr_channels);
  data->get_frequencies(frequencies, image_size, channel_offset);
}

void DATA_get_uvw(idg::Data* data, void* ptr, unsigned int nr_baselines,
                  unsigned int nr_timesteps, unsigned int baseline_offset,
                  unsigned int time_offset, float integration_time) {
  idg::Array2D<idg::UVW<float>> uvw((idg::UVW<float>*)ptr, nr_baselines,
                                    nr_timesteps);
  data->get_uvw(uvw, baseline_offset, time_offset, integration_time);
}

void DATA_print_info(idg::Data* data) { data->print_info(); }

}  // end extern "C"
