// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/Types.h"

#include <fftw3.h>

namespace idg {
namespace kernel {
namespace cpu {
namespace reference {

void kernel_gridder(KERNEL_GRIDDER_ARGUMENTS);

void kernel_degridder(KERNEL_DEGRIDDER_ARGUMENTS);

void kernel_fft(KERNEL_FFT_ARGUMENTS);

void kernel_adder(KERNEL_ADDER_ARGUMENTS);

void kernel_splitter(KERNEL_SPLITTER_ARGUMENTS);

void kernel_average_beam(KERNEL_AVERAGE_BEAM_ARGUMENTS);

}  // end namespace reference
}  // end namespace cpu
}  // end namespace kernel
}  // end namespace idg