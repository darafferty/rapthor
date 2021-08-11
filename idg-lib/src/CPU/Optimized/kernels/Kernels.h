// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/Types.h"

#include <fftw3.h>

namespace idg {
namespace kernel {
namespace cpu {
namespace optimized {

void kernel_gridder(KERNEL_GRIDDER_ARGUMENTS);

void kernel_degridder(KERNEL_DEGRIDDER_ARGUMENTS);

void kernel_fft(KERNEL_FFT_ARGUMENTS);

void kernel_adder(KERNEL_ADDER_ARGUMENTS);

void kernel_splitter(KERNEL_SPLITTER_ARGUMENTS);

/*
 * Calibration
 */
void kernel_calibrate(
    const unsigned int nr_subgrids, const unsigned long grid_size,
    const unsigned int subgrid_size, const float image_size,
    const float w_step_in_lambda, const float* __restrict__ shift,
    const unsigned int max_nr_timesteps, const unsigned int nr_channels,
    const unsigned int nr_stations, const unsigned int nr_terms,
    const unsigned int nr_time_slots, const idg::UVW<float>* uvw,
    const float* wavenumbers, std::complex<float>* visibilities,
    const float* weights, const std::complex<float>* aterms,
    const std::complex<float>* aterm_derivatives, const int* aterms_indices,
    const idg::Metadata* metadata, const std::complex<float>* subgrid,
    const std::complex<float>* phasors, double* hessian, double* gradient,
    double* residual);

void kernel_phasor(const int nr_subgrids, const long grid_size,
                   const int subgrid_size, const float image_size,
                   const float w_step_in_lambda,
                   const float* __restrict__ shift, const int max_nr_timesteps,
                   const int nr_channels, const idg::UVW<float>* uvw,
                   const float* wavenumbers, const idg::Metadata* metadata,
                   std::complex<float>* phasors);

/*
 * W-Stacking
 */
void kernel_adder_wstack(KERNEL_ADDER_WSTACK_ARGUMENTS);

void kernel_splitter_wstack(KERNEL_SPLITTER_WSTACK_ARGUMENTS);

/*
 * W-Tiling
 */
void kernel_adder_wtiles_to_grid(int grid_size, int subgrid_size,
                                 int wtile_size, float image_size, float w_step,
                                 const float* shift, int nr_tiles,
                                 const int* tile_ids,
                                 const idg::Coordinate* tile_coordinates,
                                 std::complex<float>* tiles,
                                 std::complex<float>* grid);

void kernel_adder_subgrids_to_wtiles(
    const long nr_subgrids, const int grid_size, const int subgrid_size,
    const int wtile_size, const idg::Metadata* metadata,
    const std::complex<float>* subgrid, std::complex<float>* tiles);

void kernel_splitter_wtiles_from_grid(int grid_size, int subgrid_size,
                                      int wtile_size, float image_size,
                                      float w_step, const float* shift,
                                      int nr_tiles, const int* tile_ids,
                                      const idg::Coordinate* tile_coordinates,
                                      std::complex<float>* tiles,
                                      const std::complex<float>* grid);

void kernel_splitter_subgrids_from_wtiles(
    const long nr_subgrids, const int grid_size, const int subgrid_size,
    const int wtile_size, const idg::Metadata* metadata,
    std::complex<float>* subgrid, const std::complex<float>* tiles);

}  // end namespace optimized

namespace reference {

void kernel_average_beam(KERNEL_AVERAGE_BEAM_ARGUMENTS);

}  // end namespace reference
}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg