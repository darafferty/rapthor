// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_KERNELS_CPU_H_
#define IDG_KERNELS_CPU_H_

#include <cstdint>
#include <ccomplex>
#include <cmath>
#include <string>
#include <memory>  // unique_ptr

#include "idg-common.h"

namespace idg {
namespace kernel {
namespace cpu {

class InstanceCPU : public KernelsInstance {
 public:
  static constexpr int kWTileSize = 128;

  // Constructor
  InstanceCPU();

  // Destructor
  virtual ~InstanceCPU();

/*
 * Main kernels
 */
#define KERNEL_GRIDDER_ARGUMENTS                                               \
  const int nr_subgrids, const int nr_polarizations, const long grid_size,     \
      const int subgrid_size, const float image_size,                          \
      const float w_step_in_lambda, const float *__restrict__ shift,           \
      const int nr_channels, const int nr_correlations, const int nr_stations, \
      const idg::UVW<float> *uvw, const float *wavenumbers,                    \
      const std::complex<float> *visibilities, const float *spheroidal,        \
      const std::complex<float> *aterms, const int *aterms_indices,            \
      const std::complex<float> *avg_aterm, const idg::Metadata *metadata,     \
      std::complex<float> *subgrid
  virtual void run_gridder(KERNEL_GRIDDER_ARGUMENTS) = 0;

#define KERNEL_DEGRIDDER_ARGUMENTS                                             \
  const int nr_subgrids, const int nr_polarizations, const long grid_size,     \
      const int subgrid_size, const float image_size,                          \
      const float w_step_in_lambda, const float *__restrict__ shift,           \
      const int nr_channels, const int nr_correlations, const int nr_stations, \
      const idg::UVW<float> *uvw, const float *wavenumbers,                    \
      std::complex<float> *visibilities, const float *spheroidal,              \
      const std::complex<float> *aterms, const int *aterms_indices,            \
      const idg::Metadata *metadata, const std::complex<float> *subgrid
  virtual void run_degridder(KERNEL_DEGRIDDER_ARGUMENTS) = 0;

#define KERNEL_FFT_ARGUMENTS \
  long grid_size, long size, long batch, std::complex<float> *data, int sign
  virtual void run_fft(KERNEL_FFT_ARGUMENTS) = 0;

#define KERNEL_SUBGRID_FFT_ARGUMENTS \
  long grid_size, long size, long batch, std::complex<float> *data, int sign
  virtual void run_subgrid_fft(KERNEL_SUBGRID_FFT_ARGUMENTS) = 0;

#define KERNEL_ADDER_ARGUMENTS                                              \
  const long nr_subgrids, const int nr_polarizations, const long grid_size, \
      const int subgrid_size, const idg::Metadata *metadata,                \
      const std::complex<float> *subgrid, std::complex<float> *grid
  virtual void run_adder(KERNEL_ADDER_ARGUMENTS) = 0;

#define KERNEL_SPLITTER_ARGUMENTS                                           \
  const long nr_subgrids, const int nr_polarizations, const long grid_size, \
      const int subgrid_size, const idg::Metadata *metadata,                \
      std::complex<float> *subgrid, const std::complex<float> *grid
  virtual void run_splitter(KERNEL_SPLITTER_ARGUMENTS) = 0;

#define KERNEL_AVERAGE_BEAM_ARGUMENTS                                    \
  const unsigned int nr_baselines, const unsigned int nr_antennas,       \
      const unsigned int nr_timesteps, const unsigned int nr_channels,   \
      const unsigned int nr_aterms, const unsigned int subgrid_size,     \
      const unsigned int nr_polarizations, const idg::UVW<float> *uvw,   \
      const idg::Baseline *baselines, const std::complex<float> *aterms, \
      const unsigned int *aterms_offsets, const float *weights,          \
      std::complex<float> *average_beam
  virtual void run_average_beam(KERNEL_AVERAGE_BEAM_ARGUMENTS) = 0;

/*
 * Calibration
 */
#define KERNEL_CALIBRATE_ARGUMENTS                                             \
  const unsigned int nr_subgrids, const unsigned int nr_polarizations,         \
      const unsigned long grid_size, const unsigned int subgrid_size,          \
      const float image_size, const float w_step_in_lambda,                    \
      const float *__restrict__ shift, const unsigned int max_nr_timesteps,    \
      const unsigned int nr_channels, const unsigned int nr_terms,             \
      const unsigned int nr_stations, const unsigned int nr_time_slots,        \
      const idg::UVW<float> *uvw, const float *wavenumbers,                    \
      std::complex<float> *visibilities, const float *weights,                 \
      const std::complex<float> *aterms,                                       \
      const std::complex<float> *aterm_derivatives, const int *aterms_indices, \
      const idg::Metadata *metadata, const std::complex<float> *subgrid,       \
      const std::complex<float> *phasors, double *hessian, double *gradient,   \
      double *residual
  virtual void run_calibrate(KERNEL_CALIBRATE_ARGUMENTS){};

#define KERNEL_CALIBRATE_PHASOR_ARGUMENTS                              \
  const int nr_subgrids, const long grid_size, const int subgrid_size, \
      const float image_size, const float w_step_in_lambda,            \
      const float *__restrict__ shift, const int max_nr_timesteps,     \
      const int nr_channels, const idg::UVW<float> *uvw,               \
      const float *wavenumbers, const idg::Metadata *metadata,         \
      std::complex<float> *phasors
  virtual void run_calibrate_phasor(KERNEL_CALIBRATE_PHASOR_ARGUMENTS){};

  /*
   * W-Stacking
   */
  virtual bool do_supports_wstacking() { return false; };

#define KERNEL_ADDER_WSTACK_ARGUMENTS                                      \
  int nr_subgrids, int nr_polarizations, long grid_size, int subgrid_size, \
      const idg::Metadata *metadata, const std::complex<float> *subgrid,   \
      std::complex<float> *grid
  virtual void run_adder_wstack(KERNEL_ADDER_WSTACK_ARGUMENTS){};

#define KERNEL_SPLITTER_WSTACK_ARGUMENTS                                   \
  int nr_subgrids, int nr_polarizations, long grid_size, int subgrid_size, \
      const idg::Metadata *metadata, std::complex<float> *subgrid,         \
      const std::complex<float> *grid
  virtual void run_splitter_wstack(KERNEL_SPLITTER_WSTACK_ARGUMENTS){};

  /*
   * W-Tiling
   */
  virtual bool do_supports_wtiling() { return false; };

#define KERNEL_ADDER_TILES_TO_GRID_ARGUMENTS                               \
  int nr_polarizations, int grid_size, int subgrid_size, float image_size, \
      float w_step, const float *shift, int nr_tiles, int *tile_ids,       \
      const idg::Coordinate *tile_coordinates, std::complex<float> *grid
  virtual void run_adder_tiles_to_grid(KERNEL_ADDER_TILES_TO_GRID_ARGUMENTS){};

#define KERNEL_ADDER_WTILES_ARGUMENTS                                         \
  int nr_subgrids, int nr_polarizations, int grid_size, int subgrid_size,     \
      float image_size, float w_step, const float *shift, int subgrid_offset, \
      WTileUpdateSet &wtile_flush_set, const idg::Metadata *metadata,         \
      const std::complex<float> *subgrid, std::complex<float> *grid
  virtual void run_adder_wtiles(KERNEL_ADDER_WTILES_ARGUMENTS){};

#define KERNEL_SPLITTER_WTILES_ARGUMENTS                                      \
  int nr_subgrids, int nr_polarizations, int grid_size, int subgrid_size,     \
      float image_size, float w_step, const float *shift, int subgrid_offset, \
      WTileUpdateSet &wtile_initialize_set, const idg::Metadata *metadata,    \
      std::complex<float> *subgrid, const std::complex<float> *grid
  virtual void run_splitter_wtiles(KERNEL_SPLITTER_WTILES_ARGUMENTS){};

  /**
   * Creates the buffer to store the wtiles
   *
   * The size of buffer in number of wtiles is determined by a heuristic based
   * on grid_size and wtile size kWTileSize.
   *
   * @param nr_polarizations number of polarizations in the grid
   * @param grid_size size of the grid
   * @param subgrid_size size of the subgrids
   * @return The number of wtiles
   */
  virtual size_t init_wtiles(int nr_polarizations, size_t grid_size,
                             int subgrid_size) {
    return 0;
  };

 protected:
  idg::Array1D<std::complex<float>> m_wtiles_buffer;
};

}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg

#endif
