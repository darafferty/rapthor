// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_REFERENCE_KERNELS_H_
#define IDG_REFERENCE_KERNELS_H_

#include "../common/InstanceCPU.h"

namespace idg {
namespace kernel {
namespace cpu {

class ReferenceKernels : public InstanceCPU {
  virtual void run_gridder(KERNEL_GRIDDER_ARGUMENTS) override;

  virtual void run_degridder(KERNEL_DEGRIDDER_ARGUMENTS) override;

  virtual void run_fft(KERNEL_FFT_ARGUMENTS) override;

  virtual void run_subgrid_fft(KERNEL_SUBGRID_FFT_ARGUMENTS) override;

  virtual void run_adder(KERNEL_ADDER_ARGUMENTS) override;

  virtual void run_splitter(KERNEL_SPLITTER_ARGUMENTS) override;

  virtual void run_average_beam(KERNEL_AVERAGE_BEAM_ARGUMENTS) override;
};

}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg

#endif