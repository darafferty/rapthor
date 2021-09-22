// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <iostream>
#include <cassert>    // assert
#include <algorithm>  // max_element
#include <memory.h>   // memcpy

#include "Plan.h"

using namespace std;

namespace idg {

Plan::Plan(const int kernel_size, const int subgrid_size, const int grid_size,
           const float cell_size, const Array1D<float>& shift,
           const Array1D<float>& frequencies, const Array2D<UVW<float>>& uvw,
           const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
           const Array1D<unsigned int>& aterms_offsets, Options options)
    : m_subgrid_size(subgrid_size),
      m_cell_size(cell_size),
      use_wtiles(false),
      m_options(options) {
#if defined(DEBUG)
  cout << "Plan::" << __func__ << endl;
#endif

  WTiles dummy_wtiles;
  m_shift(0) = shift(0);
  m_shift(1) = shift(1);

  initialize(kernel_size, subgrid_size, grid_size, cell_size, frequencies, uvw,
             baselines, aterms_offsets, dummy_wtiles, options);
}

Plan::Plan(const int kernel_size, const int subgrid_size, const int grid_size,
           const float cell_size, const Array1D<float>& shift,
           const Array1D<float>& frequencies, const Array2D<UVW<float>>& uvw,
           const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
           const Array1D<unsigned int>& aterms_offsets, WTiles& wtiles,
           Options options)
    : m_subgrid_size(subgrid_size),
      m_cell_size(cell_size),
      use_wtiles(true),
      m_options(options) {
#if defined(DEBUG)
  cout << "Plan::" << __func__ << " (with WTiles)" << endl;
#endif

  m_shift(0) = shift(0);
  m_shift(1) = shift(1);
  initialize(kernel_size, subgrid_size, grid_size, cell_size, frequencies, uvw,
             baselines, aterms_offsets, wtiles, options);
}

class Subgrid {
 public:
  Subgrid(const int kernel_size, const int subgrid_size, const int grid_size,
          const float w_step, const unsigned nr_w_layers, const int wtile_size)
      : kernel_size(kernel_size),
        subgrid_size(subgrid_size),
        grid_size(grid_size),
        w_step(w_step),
        nr_w_layers(nr_w_layers),
        wtile_size(wtile_size) {
    reset();
  }

  void reset() {
    u_min = std::numeric_limits<float>::infinity();
    u_max = -std::numeric_limits<float>::infinity();
    v_min = std::numeric_limits<float>::infinity();
    v_max = -std::numeric_limits<float>::infinity();
    uv_width = 0;
    w_index = 0;
    finished = false;
  }

  bool add_visibility(float u_pixels, float v_pixels, float w_lambda) {
    // Return false when finish() has been called
    if (finished) {
      return false;
    }

    // Return false for invalid visibilities
    if (std::isinf(u_pixels) || std::isinf(v_pixels)) {
      return false;
    }

    int w_index_ = 0;
    if (w_step) w_index_ = int(floorf(w_lambda));

    // if this is not the first sample, it should map to the
    // same w_index as the others, if not, return false
    if (std::isfinite(u_min) && (w_index_ != w_index)) {
      return false;
    }

    // Initialize candidate uv limits
    float u_min_ = u_min < u_pixels ? u_min : u_pixels;
    float u_max_ = u_max > u_pixels ? u_max : u_pixels;
    float v_min_ = v_min < v_pixels ? v_min : v_pixels;
    float v_max_ = v_max > v_pixels ? v_max : v_pixels;

    // Compute candidate uv width
    float u_width_ = u_max_ - u_min_;
    float v_width_ = v_max_ - v_min_;
    float uv_width_ = u_width_ > v_width_ ? u_width_ : v_width_;

    // Return false if the visibility does not fit
    if ((uv_width_ + kernel_size) >= subgrid_size) {
      return false;
    } else {
      u_min = u_min_;
      u_max = u_max_;
      v_min = v_min_;
      v_max = v_max_;
      uv_width = uv_width_;
      w_index = w_index_;
      return true;
    }
  }

  bool in_range() {
    Coordinate coordinate = get_coordinate();

    // Compute extremes of subgrid position in grid
    int uv_max_pixels = max(coordinate.x, coordinate.y);
    int uv_min_pixels = min(coordinate.x, coordinate.y);

    // Index in w-stack
    int w_index = coordinate.z;

    // Return whether the subgrid fits in grid and w-stack
    return uv_min_pixels >= 1 && uv_max_pixels <= (grid_size - subgrid_size) &&
           w_index >= -((int)nr_w_layers) && w_index < ((int)nr_w_layers);
  }

  void compute_coordinate() {
    // Compute middle point in pixels
    int u_pixels = roundf((u_max + u_min) / 2);
    int v_pixels = roundf((v_max + v_min) / 2);

    int wtile_x = floorf(float(u_pixels) / wtile_size);
    int wtile_y = floorf(float(v_pixels) / wtile_size);

    // Shift center from middle of grid to top left
    u_pixels += (grid_size / 2);
    v_pixels += (grid_size / 2);

    // Shift from middle of subgrid to top left
    u_pixels -= (subgrid_size / 2);
    v_pixels -= (subgrid_size / 2);

    coordinate = {u_pixels, v_pixels, w_index};
    wtile_coordinate = {wtile_x, wtile_y, w_index};
  }

  void finish() {
    finished = true;
    compute_coordinate();
  }

  Coordinate get_coordinate() {
    if (!finished) {
      throw std::runtime_error(
          "finish the subgrid before retrieving its coordinate");
    }
    return coordinate;
  }

  Coordinate get_wtile_coordinate() {
    if (!finished) {
      throw std::runtime_error(
          "finish the subgrid before retrieving its coordinate");
    }
    return wtile_coordinate;
  }

  const int kernel_size;
  const int subgrid_size;
  const int grid_size;
  float u_min;
  float u_max;
  float v_min;
  float v_max;
  float uv_width;
  int w_index;
  float w_step;
  int nr_w_layers;
  int wtile_size;
  bool finished;
  Coordinate coordinate;
  Coordinate wtile_coordinate;
};  // end class Subgrid

inline float meters_to_pixels(float meters, float imagesize, float frequency) {
  const double speed_of_light = 299792458.0;
  return meters * imagesize * (frequency / speed_of_light);
}

inline float meters_to_lambda(float meters, float frequency) {
  const double speed_of_light = 299792458.0;
  return meters * (frequency / speed_of_light);
}

std::vector<std::pair<int, int>> make_channel_groups(
    float baseline_length, float uv_span_frequency, float image_size,
    const Array1D<float>& frequencies, unsigned int max_nr_channels = 0) {
  std::vector<std::pair<int, int>> result;

  unsigned int nr_channels = frequencies.get_x_dim();

  // There will be at most as many channel_groups as channels
  result.reserve(nr_channels);

  float begin_pos =
      meters_to_pixels(baseline_length, image_size, frequencies(0));

  for (unsigned int begin_channel = 0; begin_channel < nr_channels;) {
    float end_pos;
    unsigned int end_channel;
    for (end_channel = begin_channel + 1; end_channel < nr_channels;
         end_channel++) {
      end_pos = meters_to_pixels(baseline_length, image_size,
                                 frequencies(end_channel));
      if (std::abs(begin_pos - end_pos) > uv_span_frequency) break;
      if (max_nr_channels > 0 &&
          (end_channel - begin_channel + 1) > max_nr_channels)
        break;
    }
    result.push_back({begin_channel, end_channel});
    begin_channel = end_channel;
    begin_pos = end_pos;
  }
  return result;
}

void Plan::initialize(
    const int kernel_size, const int subgrid_size, const int grid_size,
    const float cell_size, const Array1D<float>& frequencies,
    const Array2D<UVW<float>>& uvw,
    const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
    const Array1D<unsigned int>& aterms_offsets, WTiles& wtiles,
    const Options& options) {
#if defined(DEBUG)
  cout << "Plan::" << __func__ << endl;
  std::clog << "kernel_size  : " << kernel_size << std::endl;
  std::clog << "subgrid_size : " << subgrid_size << std::endl;
  std::clog << "grid_size    : " << grid_size << std::endl;
#endif

  // Check arguments
  assert(baselines.get_x_dim() == uvw.get_y_dim());

  // Initialize arguments
  auto nr_baselines = uvw.get_y_dim();
  auto nr_timesteps = uvw.get_x_dim();
  auto nr_timeslots = aterms_offsets.get_x_dim() - 1;
  auto nr_channels = frequencies.get_x_dim();
  auto image_size = cell_size * grid_size;  // TODO: remove
  auto wtile_size = wtiles.get_wtile_size();

  // Get options
  m_w_step = options.w_step;
  int nr_w_layers = options.nr_w_layers;
  int max_nr_timesteps_per_subgrid =
      min(options.max_nr_timesteps_per_subgrid, nr_timesteps);
  int max_nr_channels_per_subgrid = options.max_nr_channels_per_subgrid;
  bool plan_strict = options.plan_strict;

  // Spectral-line imaging
  bool simulate_spectral_line = options.simulate_spectral_line;
  auto nr_channels_ = nr_channels;
  if (simulate_spectral_line) {
    nr_channels = 1;
  }

  // Temporary metadata vector for individual baselines
  int max_nr_subgrids_per_baseline =
      max_nr_timesteps_per_subgrid > 0
          ? (nr_timesteps / max_nr_timesteps_per_subgrid) + 1
          : nr_timesteps;
  idg::Array2D<Metadata> metadata_(nr_baselines,
                                   nr_channels * max_nr_subgrids_per_baseline);

  // Count the actual number of subgrids per baseline
  std::vector<unsigned int> nr_subgrids_per_baseline(nr_baselines);

// Iterate all baselines
#pragma omp parallel for
  for (unsigned bl = 0; bl < nr_baselines; bl++) {
    // Get baseline
    unsigned int antenna1 = baselines(bl).first;
    unsigned int antenna2 = baselines(bl).second;
    Baseline baseline = (Baseline){antenna1, antenna2};

    // If the baseline is not valid continue with next baseline
    if (antenna1 == antenna2) continue;

    // Increment time_offset0 until a valid value is found
    unsigned int time_offset0 = 0;
    for (; time_offset0 < nr_timesteps; time_offset0++) {
      if (std::isfinite(uvw(bl, time_offset0).u)) break;
    }
    // If no valid value is found continue with next baseline
    if (time_offset0 == nr_timesteps) continue;

    // Reserved space for coverage in frequency from the subgrid pixel budget
    // The available pixels (subgrid_size - kernel_size) are diveded over
    // coverage in time and in frequency. The number of channels in a
    // channel_group is fixed below The remaining space is dynamically filled by
    // adding samples over time The 2.0/3.0 is a fudge factor
    float uv_frequency_span = (subgrid_size - kernel_size) * 2.0 / 3.0;

    float u = uvw(bl, time_offset0).u;
    float v = uvw(bl, time_offset0).v;
    float w = uvw(bl, time_offset0).w;

    float baseline_length = std::sqrt(u * u + v * v + w * w);

    std::vector<std::pair<int, int>> channel_groups =
        make_channel_groups(baseline_length, uv_frequency_span, image_size,
                            frequencies, max_nr_channels_per_subgrid);

    // Compute uv coordinates in pixels
    struct DataPoint {
      unsigned timestep;
      float u_pixels;
      float v_pixels;
      float w_lambda;
    };

    // Allocate datapoints for first and last channel in a group
    idg::Array2D<DataPoint> datapoints(nr_timesteps, 2);

    for (auto channel_group : channel_groups) {
      auto channel_begin = channel_group.first;
      auto channel_end = channel_group.second;

      // Initialize subgrid
      Subgrid subgrid(kernel_size, subgrid_size, grid_size, m_w_step,
                      nr_w_layers, wtile_size);

      // Constants over nr_timesteps
      const double speed_of_light = 299792458.0;
      const float frequency_begin = frequencies(channel_begin);
      const float frequency_end = frequencies(channel_end - 1);
      const float scale_begin = frequency_begin / speed_of_light;
      const float scale_end = frequency_end / speed_of_light;
      const float scale_w = 1.0f / m_w_step;

      for (unsigned t = 0; t < nr_timesteps; t++) {
        // U,V in meters
        float u_meters = uvw(bl, t).u;
        float v_meters = uvw(bl, t).v;
        float w_meters = uvw(bl, t).w;

        // U,V,W for first channel
        float u_pixels_begin = u_meters * image_size * scale_begin;
        float v_pixels_begin = v_meters * image_size * scale_begin;
        float w_lambda_begin = w_meters * scale_begin * scale_w;

        // U,V,W for last channel
        float u_pixels_end = u_meters * image_size * scale_end;
        float v_pixels_end = v_meters * image_size * scale_end;
        float w_lambda_end = 0;  // not used

        datapoints(t, 0) = {t, u_pixels_begin, v_pixels_begin, w_lambda_begin};
        datapoints(t, 1) = {t, u_pixels_end, v_pixels_end, w_lambda_end};
      }  // end for time

      unsigned int time_offset = time_offset0;
      while (time_offset < nr_timesteps) {
        // Create subgrid
        subgrid.reset();
        int nr_timesteps_subgrid = 0;

        // Load first visibility
        DataPoint first_datapoint = datapoints(time_offset, 0);
        const int first_timestep = first_datapoint.timestep;

        // Iterate all datapoints
        for (; time_offset < nr_timesteps; time_offset++) {
          // Visibility for first channel
          DataPoint visibility0 = datapoints(time_offset, 0);
          const float u_pixels0 = visibility0.u_pixels;
          const float v_pixels0 = visibility0.v_pixels;
          const float w_lambda0 = visibility0.w_lambda;

          // Visibility for last channel
          DataPoint visibility1 = datapoints(time_offset, 1);
          const float u_pixels1 = visibility1.u_pixels;
          const float v_pixels1 = visibility1.v_pixels;

          // Try to add visibilities to subgrid
          if (subgrid.add_visibility(u_pixels0, v_pixels0, w_lambda0) &&
              // HACK also pass w_lambda0 below
              subgrid.add_visibility(u_pixels1, v_pixels1, w_lambda0)) {
            nr_timesteps_subgrid++;
            if (nr_timesteps_subgrid == max_nr_timesteps_per_subgrid) break;
          } else {
            break;
          }
        }  // end for time

        // Handle empty subgrid
        if (nr_timesteps_subgrid == 0) {
          DataPoint visibility = datapoints(time_offset, 0);
          const float u_pixels = visibility.u_pixels;
          const float v_pixels = visibility.v_pixels;

          if (std::isfinite(u_pixels) && std::isfinite(v_pixels) &&
              plan_strict) {
// Coordinates are valid, but did not (all) fit onto subgrid
#pragma omp critical
            throw std::runtime_error(
                "could not place (all) visibilities on subgrid (too many "
                "channnels, kernel size too large)");
          } else {
            // Advance to next timeslot when visibilities for current timeslot
            // had infinite coordinates
            time_offset++;
            continue;
          }
        }

        // Compute time index for first visibility on subgrid
        auto time_index = bl * nr_timesteps + first_timestep;

        // Finish subgrid
        subgrid.finish();

        // Add subgrid to metadata
        if (subgrid.in_range()) {
          Metadata m = {
              .time_index = (int)time_index,         // time index
              .nr_timesteps = nr_timesteps_subgrid,  // nr of timesteps
              .channel_begin = channel_begin,
              .channel_end = channel_end,
              .baseline = baseline,                    // baselines
              .coordinate = subgrid.get_coordinate(),  // coordinate
              .wtile_coordinate =
                  subgrid.get_wtile_coordinate(),  // tile coordinate
              .wtile_index = -1,  // tile index, to be filled in combine step
              .nr_aterms = -1     // nr of aterms, to be filled in later
          };

          unsigned subgrid_idx = nr_subgrids_per_baseline[bl];
          metadata_(bl, subgrid_idx++) = m;

          // Add additional subgrids for subsequent frequencies
          if (simulate_spectral_line) {
            for (unsigned c = 1; c < nr_channels_; c++) {
              // Compute shifted subgrid for current frequency
              float shift = frequencies(c) / frequencies(0);
              Metadata m = metadata_(bl, subgrid_idx);
              m.coordinate.x *= shift;
              m.coordinate.y *= shift;
              metadata_(bl, subgrid_idx++) = m;
            }
          }

          // Update number of subgrids
          nr_subgrids_per_baseline[bl] = subgrid_idx;
        } else if (plan_strict) {
#pragma omp critical
          {
            Coordinate coordinate = subgrid.get_coordinate();
            std::stringstream message;
            message << "subgrid out of range: "
                    << "coordinate = (" << coordinate.x << ", " << coordinate.y
                    << ")";
            std::cout << message.str() << std::endl;
            throw std::runtime_error(message.str());
          }
        }
      }  // end while
    }    // end for channel_groups
  }      // end for bl

  // Find the total number of subgrids for all baselines
  int total_nr_subgrids = std::accumulate(nr_subgrids_per_baseline.begin(),
                                          nr_subgrids_per_baseline.end(), 0);

  // Allocate member variables
  metadata.resize(total_nr_subgrids);
  total_nr_timesteps_per_baseline.resize(nr_baselines);
  total_nr_visibilities_per_baseline.resize(nr_baselines);
  subgrid_offset.resize(nr_baselines);

  // Combine data structures
  unsigned subgrid_index = 0;
  for (unsigned bl = 0; bl < nr_baselines; bl++) {
    // The subgrid offset is the number of subgrids for all prior baselines
    subgrid_offset[bl] = subgrid_index;

    // Count total number of timesteps for baseline
    int total_nr_timesteps = 0;

    for (unsigned int i = 0; i < nr_subgrids_per_baseline[bl]; i++) {
      Metadata& m = metadata_(bl, i);

      // Set wtile_index
      Coordinate& wtile_coordinate = m.wtile_coordinate;
      m.wtile_index = wtiles.add_subgrid(subgrid_index, wtile_coordinate);

      // Append subgrid
      metadata[subgrid_index++] = m;

      // Accumulate timesteps, taking only the
      // first channel group into account
      if (m.channel_begin == 0) {
        total_nr_timesteps += m.nr_timesteps;
      }
    }

    // Set total total number of timesteps for baseline
    total_nr_timesteps_per_baseline[bl] = total_nr_timesteps;

    // Either all or no channels of a timestep are gridded
    // onto a subgrid, hence total_nr_timesteps * nr_channels
    int total_nr_visibilities = total_nr_timesteps * nr_channels;
    total_nr_visibilities_per_baseline[bl] = total_nr_visibilities;
  }  // end for bl

  // Reserve aterm indices
  aterm_indices.resize(nr_baselines * nr_timesteps);

// Set aterm index for every timestep
#pragma omp parallel for
  for (unsigned bl = 0; bl < nr_baselines; bl++) {
    unsigned time_idx = 0;

    for (unsigned timeslot = 0; timeslot < nr_timeslots; timeslot++) {
      // Get aterm offset
      const unsigned current_aterms_offset = aterms_offsets(timeslot);
      const unsigned next_aterms_offset = aterms_offsets(timeslot + 1);

      // The aterm index is equal to the timeslot
      const unsigned aterm_index = timeslot;

      // Determine number of timesteps in current aterm
      const unsigned nr_timesteps_per_aterm =
          next_aterms_offset - current_aterms_offset;

      for (unsigned timestep = 0; timestep < nr_timesteps_per_aterm;
           timestep++) {
        aterm_indices[bl * nr_timesteps + time_idx++] = aterm_index;
      }
    }
  }

  // Set nr_aterms
#pragma omp parallel for
  for (unsigned i = 0; i < metadata.size(); i++) {
    auto& m = metadata[i];
    auto aterm_index = aterm_indices[m.time_index];
    auto nr_aterms = 1;
    for (auto time = 0; time < m.nr_timesteps; time++) {
      auto aterm_index_current = aterm_indices[m.time_index + time];
      if (aterm_index != aterm_index_current) {
        nr_aterms++;
        aterm_index = aterm_index_current;
      }
    }

    m.nr_aterms = nr_aterms;
  }

  // Set sentinel
  subgrid_offset.push_back(metadata.size());

  m_wtile_initialize_set = wtiles.get_initialize_set();
  m_wtile_flush_set = wtiles.get_flush_set();

#if defined(DEBUG)
  std::clog << "nr_baselines    : " << nr_baselines << " (input)" << std::endl;
  std::clog << "nr_timesteps    : " << nr_timesteps << " (per baseline)"
            << std::endl;
  std::clog << "nr_channels     : " << nr_channels << " (per baseline)"
            << std::endl;
  std::clog << "nr_visibilities : " << get_nr_visibilities() << " (planned)"
            << std::endl;
  std::clog << "nr_subgrids     : " << get_nr_subgrids() << " (planned)"
            << std::endl;
#endif
}  // end initialize

int Plan::get_nr_subgrids() const { return metadata.size(); }

int Plan::get_nr_subgrids(int bl) const { return get_nr_subgrids(bl, 1); }

int Plan::get_nr_subgrids(int bl, int n) const {
  if (n < 1) {
    throw invalid_argument("n should be at least one.");
  }
  return get_subgrid_offset(bl + n) - get_subgrid_offset(bl);
}

int Plan::get_subgrid_offset(int bl) const { return subgrid_offset[bl]; }

int Plan::get_max_nr_subgrids(int bl1, int bl2, int n) const {
  int nr_baselines = bl1 + n > bl2 ? bl2 - bl1 : n;
  int max_nr_subgrids = get_nr_subgrids(bl1, nr_baselines);
  for (int bl = bl1 + n; bl < bl2; bl += n) {
    nr_baselines = bl + n > bl2 ? bl2 - bl : n;
    int nr_subgrids = get_nr_subgrids(bl, nr_baselines);
    if (nr_subgrids > max_nr_subgrids) {
      max_nr_subgrids = nr_subgrids;
    }
  }
  return max_nr_subgrids;
}

int Plan::get_max_nr_subgrids(int n) const {
  return get_max_nr_subgrids(0, get_nr_baselines(), n);
}

int Plan::get_nr_timesteps() const {
  return accumulate(total_nr_timesteps_per_baseline.begin(),
                    total_nr_timesteps_per_baseline.end(), 0);
}

int Plan::get_nr_timesteps(int baseline) const {
  return total_nr_timesteps_per_baseline[baseline];
}

int Plan::get_nr_timesteps(int baseline, int n) const {
  assert(n <= int(total_nr_timesteps_per_baseline.size()) - baseline);
  auto begin = next(total_nr_timesteps_per_baseline.begin(), baseline);
  auto end = next(begin, n);
  return accumulate(begin, end, 0);
}

int Plan::get_max_nr_timesteps_subgrid() const {
  if (!metadata.size()) return 0;
  auto max_nr_timesteps = metadata[0].nr_timesteps;
  for (const Metadata& m : metadata) {
    if (m.nr_timesteps > max_nr_timesteps) {
      max_nr_timesteps = m.nr_timesteps;
    }
  }
  return max_nr_timesteps;
}

int Plan::get_nr_visibilities() const {
  return accumulate(total_nr_visibilities_per_baseline.begin(),
                    total_nr_visibilities_per_baseline.end(), 0);
}

int Plan::get_nr_visibilities(int baseline) const {
  return total_nr_visibilities_per_baseline[baseline];
}

int Plan::get_nr_visibilities(int baseline, int n) const {
  auto begin = next(total_nr_visibilities_per_baseline.begin(), baseline);
  auto end = next(begin, n);
  return accumulate(begin, end, 0);
}

const Metadata* Plan::get_metadata_ptr(int bl) const {
  auto offset = get_subgrid_offset(bl);
  return &(metadata[offset]);
}

size_t Plan::get_sizeof_metadata() const {
  return get_nr_subgrids() * sizeof(idg::Metadata);
}

void Plan::copy_metadata(void* ptr) const {
  memcpy(ptr, get_metadata_ptr(), get_nr_subgrids() * sizeof(Metadata));
}

const int* Plan::get_aterm_indices_ptr(int bl) const {
  auto offset = get_subgrid_offset(bl);
  return &(aterm_indices[offset]);
}

void Plan::initialize_job(const unsigned int nr_baselines,
                          const unsigned int jobsize, const unsigned int bl,
                          unsigned int* first_bl_, unsigned int* last_bl_,
                          unsigned int* current_nr_baselines_) const {
  // Determine maximum number of baselines in this job
  auto current_nr_baselines =
      bl + jobsize > nr_baselines ? nr_baselines - bl : jobsize;

  // Determine first and last baseline in this job
  auto first_bl = bl;
  auto last_bl = bl + current_nr_baselines;

  // Skip empty baselines
  while (first_bl < last_bl && get_nr_timesteps(first_bl, 1) == 0) {
    first_bl++;
  }

  // Update parameters
  (*first_bl_) = first_bl;
  (*last_bl_) = last_bl;
  (*current_nr_baselines_) = last_bl - first_bl;
}

void Plan::mask_visibilities(Array5D<std::complex<float>>& visibilities) const {
  // Get visibilities dimensions
  auto nr_baselines = visibilities.get_d_dim();
  auto nr_timesteps = visibilities.get_c_dim();
  auto nr_channels = visibilities.get_b_dim();
  auto nr_correlations = visibilities.get_a_dim();

  // The visibility mask is zero
  const std::complex<float> zero = {0.0f, 0.0f};

  // Sanity check
  assert((unsigned)get_nr_baselines() == nr_baselines);

  // Iterate all metadata elements
  int nr_subgrids = get_nr_subgrids();
  for (int i = 0; i < nr_subgrids; i++) {
    const Metadata& m_current = metadata[i];

    // Determine which visibilities are used in the plan
    unsigned time_index = m_current.time_index;
    unsigned current_nr_timesteps = m_current.nr_timesteps;

    // Determine which visibilities to mask
    unsigned first = time_index + current_nr_timesteps;
    unsigned last = 0;
    if (i < nr_subgrids - 1) {
      const Metadata& m_next = metadata[i + 1];
      int next_index = m_next.time_index;
      last = next_index;
    } else {
      last = nr_baselines * nr_timesteps;
    }

    // Mask all selected visibilities for all channels
    for (unsigned t = first; t < last; t++) {
      for (unsigned c = 0; c < nr_channels; c++) {
        for (unsigned cor = 0; cor < nr_correlations; cor++) {
          visibilities(0, t, c, cor) = zero;
        }
      }
    }
  }
}

size_t Plan::baseline_index(size_t antenna1, size_t antenna2,
                            size_t nr_stations) {
  assert(antenna1 < antenna2);
  size_t offset = antenna1 * nr_stations - ((antenna1 + 1) * antenna1) / 2 - 1;
  return antenna2 - antenna1 + offset;
}

}  // namespace idg
