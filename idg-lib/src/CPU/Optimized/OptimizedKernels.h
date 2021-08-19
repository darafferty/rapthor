// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_OPTIMIZED_KERNELS_H_
#define IDG_OPTIMIZED_KERNELS_H_

#include "../common/InstanceCPU.h"
#include "../Reference/ReferenceKernels.h"

namespace idg {
namespace kernel {
namespace cpu {

class OptimizedKernels : public InstanceCPU {
  /*
   * Main
   */
  virtual void run_gridder(KERNEL_GRIDDER_ARGUMENTS) override;

  virtual void run_degridder(KERNEL_DEGRIDDER_ARGUMENTS) override;

  virtual void run_fft(KERNEL_FFT_ARGUMENTS) override;

  virtual void run_subgrid_fft(KERNEL_SUBGRID_FFT_ARGUMENTS) override;

  virtual void run_adder(KERNEL_ADDER_ARGUMENTS) override;

  virtual void run_splitter(KERNEL_SPLITTER_ARGUMENTS) override;

  virtual void run_average_beam(KERNEL_AVERAGE_BEAM_ARGUMENTS) override;

  /*
   * Calibration
   */
  virtual void run_calibrate(KERNEL_CALIBRATE_ARGUMENTS) override;

  virtual void run_calibrate_phasor(KERNEL_CALIBRATE_PHASOR_ARGUMENTS) override;

  /*
   * W-Stacking
   */
  bool do_supports_wstacking() override { return true; };

  virtual void run_adder_wstack(KERNEL_ADDER_WSTACK_ARGUMENTS) override;

  virtual void run_splitter_wstack(KERNEL_SPLITTER_WSTACK_ARGUMENTS) override;

  /*
   * W-Tiling
   */
  bool do_supports_wtiling() override { return true; };

  virtual size_t init_wtiles(int nr_polarizations, size_t grid_size,
                             int subgrid_size) override;

  virtual void run_adder_tiles_to_grid(
      KERNEL_ADDER_TILES_TO_GRID_ARGUMENTS) override;

  virtual void run_adder_wtiles(KERNEL_ADDER_WTILES_ARGUMENTS) override;

  virtual void run_splitter_wtiles(KERNEL_SPLITTER_WTILES_ARGUMENTS) override;
};

}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg

#endif