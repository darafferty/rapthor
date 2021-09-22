// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_PLAN_H_
#define IDG_PLAN_H_

#include <vector>
#include <limits>
#include <stdexcept>  // runtime_error
#include <cmath>
#include <numeric>
#include <iterator>
#include <omp.h>

#include "Types.h"
#include "ArrayTypes.h"
#include "WTiles.h"

namespace idg {

// forward declaration of friend classes
// The Plan constructors are private
// only these classes can instantiate a Plan
namespace proxy {
class Proxy;
namespace cpu {
class CPU;
}
namespace hybrid {
class UnifiedOptimized;
}
}  // namespace proxy

class Plan {
 public:
  enum Mode { FULL_POLARIZATION, STOKES_I_ONLY };

  struct Options {
    Options() {}

    // w-stacking
    float w_step = 0.0;
    unsigned nr_w_layers = 1;

    // throw error when visibilities do not fit onto subgrid
    bool plan_strict = false;

    // limit the maximum number of timesteps per subgrid
    // zero means no limit
    unsigned max_nr_timesteps_per_subgrid = 0;

    // limit the maximum number of channels per subgrid
    // zero means no limit
    unsigned max_nr_channels_per_subgrid = 0;

    // consider only first channel when creating subgrids,
    // add additional subgrids for every subsequent frequencies
    bool simulate_spectral_line = false;

    // Imaging mode
    Mode mode = Mode::FULL_POLARIZATION;
  };

  // Constructors
  Plan(){};

  Plan(Plan&&) = default;

  Plan(const int kernel_size, const int subgrid_size, const int grid_size,
       const float cell_size, const Array1D<float>& shift,
       const Array1D<float>& frequencies, const Array2D<UVW<float>>& uvw,
       const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
       const Array1D<unsigned int>& aterms_offsets,
       Options options = Options());

  Plan(const int kernel_size, const int subgrid_size, const int grid_size,
       const float cell_size, const Array1D<float>& shift,
       const Array1D<float>& frequencies, const Array2D<UVW<float>>& uvw,
       const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
       const Array1D<unsigned int>& aterms_offsets, WTiles& wtiles,
       Options options = Options());

  // Destructor
  virtual ~Plan() = default;

  void initialize(
      const int kernel_size, const int subgrid_size, const int grid_size,
      const float cell_size, const Array1D<float>& frequencies,
      const Array2D<UVW<float>>& uvw,
      const Array1D<std::pair<unsigned int, unsigned int>>& baselines,
      const Array1D<unsigned int>& aterms_offsets, WTiles& wtiles,
      const Options& options);

  // options
  const Options get_options() const { return m_options; };
  // total number of subgrids
  int get_nr_subgrids() const;

  // number of subgrids one baseline
  int get_nr_subgrids(int baseline) const;

  // number of subgrids for baselines b1 to b1+n-1
  int get_nr_subgrids(int baseline, int n) const;

  // returns index of first index of baseline
  int get_subgrid_offset(int baseline) const;

  // max number of subgrids for n baselines between bl1 and bl2+n
  int get_max_nr_subgrids(int bl1, int bl2, int n) const;

  // max number of subgrids for n baselines
  int get_max_nr_subgrids(int n = 1) const;

  // total number of timesteps
  int get_nr_timesteps() const;

  // number of timesteps one baseline
  int get_nr_timesteps(int baseline) const;

  // number of timesteps for baselines b1 to b1+n-1
  int get_nr_timesteps(int baseline, int n) const;

  // max number of timesteps for 1 subgrid
  int get_max_nr_timesteps_subgrid() const;

  // total number of visibilities
  int get_nr_visibilities() const;

  // number of visibilities one baseline
  int get_nr_visibilities(int baseline) const;

  // number of visibilities for baselines b1 to b1+n-1
  int get_nr_visibilities(int baseline, int n) const;

  // number of baselines
  int get_nr_baselines() const {
    return total_nr_timesteps_per_baseline.size();
  }

  const Metadata* get_metadata_ptr(int baseline = 0) const;

  size_t get_sizeof_metadata() const;

  void copy_metadata(void* ptr) const;

  const int* get_aterm_indices_ptr(int baseline = 0) const;

  void initialize_job(const unsigned int nr_baselines,
                      const unsigned int jobsize, const unsigned int bl,
                      unsigned int* first_bl, unsigned int* last_bl,
                      unsigned int* current_nr_baselines) const;

  // set visibilities not used by plan to zero
  void mask_visibilities(Array5D<std::complex<float>>& visibilities) const;

  WTileUpdateSet get_wtile_initialize_set() const {
    return m_wtile_initialize_set;
  }

  WTileUpdateSet get_wtile_flush_set() const { return m_wtile_flush_set; }

  bool get_use_wtiles() const { return use_wtiles; }

  /* Creates a baseline index for use in a baselines array.
   *   0 implies antenna1=0, antenna2=1 ;
   *   1 implies antenna1=0, antenna2=2 ;
   * n-1 implies antenna1=0, antenna2=n ;
   *   n implies antenna1=1, antenna2=0 ; etc.
   */
  static size_t baseline_index(size_t antenna1, size_t antenna2,
                               size_t nr_stations);

  int get_subgrid_size() const { return m_subgrid_size; }
  float get_w_step() const { return m_w_step; }

  const Array1D<float>& get_shift() const { return m_shift; }
  float get_cell_size() const { return m_cell_size; }

 private:
  Array1D<float> m_shift{2};
  int m_subgrid_size;
  float m_w_step;
  float m_cell_size;
  std::vector<Metadata> metadata;
  std::vector<int> subgrid_offset;
  std::vector<int> total_nr_timesteps_per_baseline;
  std::vector<int> total_nr_visibilities_per_baseline;
  WTileUpdateSet m_wtile_initialize_set;
  WTileUpdateSet m_wtile_flush_set;
  std::vector<int> aterm_indices;
  bool use_wtiles;
  Options m_options;
};  // class Plan

}  // namespace idg

#endif
