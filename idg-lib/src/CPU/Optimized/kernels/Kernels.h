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
void kernel_calibrate(KERNEL_CALIBRATE_ARGUMENTS);

void kernel_phasor(KERNEL_CALIBRATE_PHASOR_ARGUMENTS);

/*
 * W-Stacking
 */
void kernel_adder_wstack(KERNEL_ADDER_WSTACK_ARGUMENTS);

void kernel_splitter_wstack(KERNEL_SPLITTER_WSTACK_ARGUMENTS);

/*
 * W-Tiling
 */
void kernel_adder_wtiles_to_grid(
    int nr_polarizations, int grid_size, int subgrid_size, int wtile_size,
    float image_size, float w_step, const float* shift, int nr_tiles,
    const int* tile_ids, const idg::Coordinate* tile_coordinates,
    std::complex<float>* tiles, std::complex<float>* grid);

void kernel_adder_subgrids_to_wtiles(
    const long nr_subgrids, const int nr_polarizations, const int grid_size,
    const int subgrid_size, const int wtile_size, const idg::Metadata* metadata,
    const std::complex<float>* subgrid, std::complex<float>* tiles);

void kernel_splitter_wtiles_from_grid(
    int nr_polarizations, int grid_size, int subgrid_size, int wtile_size,
    float image_size, float w_step, const float* shift, int nr_tiles,
    const int* tile_ids, const idg::Coordinate* tile_coordinates,
    std::complex<float>* tiles, const std::complex<float>* grid);

void kernel_splitter_subgrids_from_wtiles(
    const long nr_subgrids, const int nr_polarizations, const int grid_size,
    const int subgrid_size, const int wtile_size, const idg::Metadata* metadata,
    std::complex<float>* subgrid, const std::complex<float>* tiles);

}  // end namespace optimized

namespace reference {

void kernel_average_beam(KERNEL_AVERAGE_BEAM_ARGUMENTS);

}  // end namespace reference
}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg