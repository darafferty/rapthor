// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "math.cu"
#include "Types.h"

/*
    Kernel
*/
extern "C" {
__global__ void kernel_average_beam(
    const unsigned int                  nr_channels,
    const UVW<float>*      __restrict__ uvw,
    const Baseline*        __restrict__ baselines,
    const float2*          __restrict__ aterms,
    const int*             __restrict__ aterms_indices,
    const float*           __restrict__ weights,
          float2*          __restrict__ average_beam)
{
    // TODO
}

} // end extern "C"