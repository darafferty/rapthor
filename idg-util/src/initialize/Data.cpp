// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <random>

#include "uvwsim.h"

#include "Data.h"

/* Constants */
#define SPEED_OF_LIGHT 299792458.0

namespace idg {

Data::Data(std::string layout_file) {
  // Set station_coordinates
  set_station_coordinates(layout_file);

  // Set baselines
  set_baselines(m_station_coordinates);
}

void Data::set_station_coordinates(std::string layout_file = "SKA1_low_ecef") {
  // Check whether layout file exists
  std::string filename = std::string(IDG_DATA_DIR) + "/" + layout_file;
  if (!uvwsim_file_exists(filename.c_str())) {
    std::cerr << "Failed to find specified layout file: " << filename
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // Get number of stations
  unsigned nr_stations = uvwsim_get_num_stations(filename.c_str());
  if (nr_stations < 0) {
    std::cerr << "Failed to read any stations from layout file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Allocate memory for station coordinates
  double* x = (double*)malloc(nr_stations * sizeof(double));
  double* y = (double*)malloc(nr_stations * sizeof(double));
  double* z = (double*)malloc(nr_stations * sizeof(double));

  // Load the antenna coordinates
  unsigned nr_stations_read =
      uvwsim_load_station_coords(filename.c_str(), nr_stations, x, y, z);
  if (nr_stations_read != nr_stations) {
    std::cerr << "Failed to read antenna coordinates." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Create vector of station coordinates
  for (unsigned i = 0; i < nr_stations; i++) {
    Data::StationCoordinate coordinate = {x[i], y[i], z[i]};
    m_station_coordinates.push_back(coordinate);
  }

  // Free memory
  free(x);
  free(y);
  free(z);
}

void Data::print_info() {
  std::cout << "number of stations: " << m_station_coordinates.size()
            << std::endl;
  std::cout << "number of baselines: " << m_baselines.size() << std::endl;
  std::cout << "longest baseline = " << get_max_uv() * 1e-3 << " km"
            << std::endl;
}

float Data::compute_image_size(unsigned long grid_size) {
  // the origin of the grid is at the center, therefore any baseline
  // should fit within half of the diameter of the grid
  grid_size /= (2 * grid_padding);
  auto max_uv = get_max_uv();
  return grid_size / max_uv * (SPEED_OF_LIGHT / start_frequency);
}

float Data::compute_max_uv(unsigned long grid_size) {
  float fov_arcsec = fov_deg * 3600;
  float wavelength = SPEED_OF_LIGHT / start_frequency;
  float res_arcsec = fov_arcsec / grid_size;
  float max_uv = (180 * 3600) * weight * wavelength / (M_PI * res_arcsec);
  return max_uv;
}

unsigned int Data::compute_grid_size() {
  float max_uv = get_max_uv();
  float fov_arcsec = fov_deg * 3600;
  float wavelength = SPEED_OF_LIGHT / start_frequency;
  float res_arcsec = ((180 * 3600) * weight * wavelength / M_PI) / max_uv;
  unsigned int grid_size = fov_arcsec / res_arcsec;
  grid_size *= grid_padding;
  return grid_size;
}

void Data::set_baselines(std::vector<StationCoordinate>& station_coordinates) {
  unsigned int nr_stations = station_coordinates.size();
  printf("nr_stations = %d, nr_baselines = %lu\n", nr_stations,
         m_baselines.size());

  // Set baselines from station pairs
  for (unsigned station1 = 0; station1 < nr_stations; station1++) {
    for (unsigned station2 = station1 + 1; station2 < nr_stations; station2++) {
      Baseline baseline = {station1, station2};
      m_baselines.push_back(std::pair<float, Baseline>(0, baseline));
    }
  }

  unsigned int nr_baselines = m_baselines.size();

// Fill in the maximum uv length (in meters)
#pragma omp parallel for
  for (unsigned int bl = 0; bl < nr_baselines; bl++) {
    Baseline baseline = m_baselines[bl].second;
    double u, v, w;

    // Compute uvw values for 24 hours of observation (with steps of 1 hours)
    float max_uv = 0.0f;
    for (unsigned time = 0; time < 24; time++) {
      float integration_time = 0.9;
      unsigned int timestep = time * 3600;
      evaluate_uvw(baseline, timestep, integration_time, &u, &v, &w);
      float baseline_length = sqrtf(u * u + v * v);
      max_uv = std::max(max_uv, baseline_length);
    }  // end for time

    // Set max_uv for current baseline
    m_baselines[bl].first = max_uv;
  }  // end for bl
}

void Data::limit_max_baseline_length(float max_uv) {
  // Select the baselines up to max_uv meters long
  std::vector<std::pair<float, Baseline>> baselines_selected;
  for (auto entry : m_baselines) {
    if (entry.first < max_uv) {
      baselines_selected.push_back(entry);
    }
  }

  // Update baselines
  std::swap(m_baselines, baselines_selected);
}

bool sort_baseline_ascending(const std::pair<float, Baseline>& a,
                             const std::pair<float, Baseline>& b) {
  return (a.first < b.first);
}

bool sort_baseline_descending(const std::pair<float, Baseline>& a,
                              const std::pair<float, Baseline>& b) {
  return (a.first > b.first);
}

void Data::limit_nr_stations(unsigned int n) {
  // The selected stations
  std::vector<StationCoordinate> stations_selected;

  // Make copy of stations
  std::vector<StationCoordinate> stations_copy = m_station_coordinates;

  // Random number generator
  std::mt19937 generator(0);

  // Select random stations
  for (unsigned i = 0; i < n; i++) {
    auto min = 0;
    auto max = stations_copy.size();
    std::uniform_int_distribution<> distribution(min, max);
    auto idx = distribution(generator);
    stations_selected.push_back(stations_copy[idx]);
    stations_copy.erase(stations_copy.begin() + idx);
  }

  // Update stations
  std::swap(m_station_coordinates, stations_selected);
}

void Data::limit_nr_baselines(unsigned int n) {
  if (n > m_baselines.size()) {
    return;
  }

  // The selected baselines
  std::vector<std::pair<float, Baseline>> baselines_selected;

  // Make copy of baselines
  std::vector<std::pair<float, Baseline>> baselines_copy = m_baselines;

  // Sort baselines on length
  std::sort(baselines_copy.begin(), baselines_copy.end(),
            sort_baseline_descending);

  // Make uniform selection of baselines
  for (unsigned i = 0; i < n; i++) {
    auto index = i * (m_baselines.size() / n);
    baselines_selected.push_back(baselines_copy[index]);
  }

  // Update baselines
  std::swap(m_baselines, baselines_selected);
}

void Data::get_frequencies(Array1D<float>& frequencies, float image_size,
                           unsigned int channel_offset) const {
  auto nr_channels = frequencies.get_x_dim();
  auto max_uv = get_max_uv();
  float frequency_increment = SPEED_OF_LIGHT / (max_uv * image_size);
  for (unsigned chan = 0; chan < nr_channels; chan++) {
    frequencies(chan) =
        start_frequency + frequency_increment * (chan + channel_offset);
  }
}

Array2D<UVW<float>> Data::get_uvw(unsigned int nr_baselines,
                                  unsigned int nr_timesteps,
                                  unsigned int baseline_offset,
                                  unsigned int time_offset,
                                  float integration_time) const {
  Array2D<UVW<float>> uvw(nr_baselines, nr_timesteps);
  get_uvw(uvw, baseline_offset, time_offset, integration_time);
  return uvw;
}

void Data::get_uvw(Array2D<UVW<float>>& uvw, unsigned int baseline_offset,
                   unsigned int time_offset, float integration_time) const {
  unsigned int nr_baselines_total = m_baselines.size();
  unsigned int nr_baselines = uvw.get_y_dim();
  unsigned int nr_timesteps = uvw.get_x_dim();

  if (baseline_offset + nr_baselines > nr_baselines_total) {
    std::cerr << "Out-of-range baselines selected: ";
    if (baseline_offset > 0) {
      std::cerr << baseline_offset << " + " << nr_baselines;
    }
    std::cerr << nr_baselines << " > " << nr_baselines_total << std::endl;
    nr_baselines = nr_baselines_total;
  }

// Evaluate uvw per baseline
#pragma omp parallel for
  for (unsigned bl = 0; bl < nr_baselines; bl++) {
    std::pair<float, Baseline> e = m_baselines[baseline_offset + bl];
    Baseline& baseline = e.second;

    for (unsigned long time = 0; time < nr_timesteps; time++) {
      double u, v, w;
      evaluate_uvw(baseline, time_offset + time, integration_time, &u, &v, &w);
      uvw(bl, time) = {(float)u, (float)v, (float)w};
    }  // end for time
  }    // end for bl
}

float Data::get_max_uv() const {
  float max_uv = 0;
  for (auto baseline : m_baselines) {
    max_uv = std::max(max_uv, baseline.first);
  }
  return max_uv;
}

void Data::evaluate_uvw(Baseline& baseline, unsigned int timestep,
                        float integration_time, double* u, double* v,
                        double* w) const {
  unsigned station1 = baseline.station1;
  unsigned station2 = baseline.station2;
  double x1 = m_station_coordinates[station1].x;
  double y1 = m_station_coordinates[station1].y;
  double z1 = m_station_coordinates[station1].z;
  double x2 = m_station_coordinates[station2].x;
  double y2 = m_station_coordinates[station2].y;
  double z2 = m_station_coordinates[station2].z;
  double x[2] = {x1, x2};
  double y[2] = {y1, y2};
  double z[2] = {z1, z2};
  unsigned long time = observation_hour * (3600) + observation_minute * 60 +
                       observation_seconds + timestep * integration_time;
  int hour = time / 3600;
  time = time % 3600;
  int minute = time / 60;
  time = time % 60;
  int seconds = time;
  double time_mjd =
      uvwsim_datetime_to_mjd(observation_year, observation_month,
                             observation_day, hour, minute, seconds);
  uvwsim_evaluate_baseline_uvw(u, v, w, 2, x, y, z, observation_ra,
                               observation_dec, time_mjd);
}

}  // namespace idg

#include "datac.h"
