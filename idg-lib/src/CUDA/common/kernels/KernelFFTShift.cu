// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cuComplex.h>

#include "Types.h"
#include "math.cu"

extern "C" {

inline __device__ size_t index(size_t n, int i,
                               int y, int x)
{
    // data: [batch][n]]n]
    return i * n * n +
           y * n + x;
}

/*
    Kernel
*/
__global__ void kernel_fft_shift(
    const long     n,
          float2*  data,
          float2   scale)
{
    unsigned long i = blockIdx.x;
    unsigned long y = blockIdx.y;

    unsigned long n2 = n / 2;

    for (unsigned long x = threadIdx.x; x < n2; x += blockDim.x)
    {
        if (y < n2)
        {
            unsigned long idx1 = index(n, i, y, x);
            unsigned long idx3 = index(n, i, y + n2, x + n2);
            float2 tmp1 = data[idx1];
            float2 tmp3 = data[idx3];
            data[idx1] = make_float2(tmp3.x * scale.x, tmp3.y * scale.y);
            data[idx3] = make_float2(tmp1.x * scale.x, tmp1.y * scale.y);

            unsigned long idx2 = index(n, i, y + n2, x);
            unsigned long idx4 = index(n, i, y, x + n2);
            float2 tmp2 = data[idx2];
            float2 tmp4 = data[idx4];
            data[idx2] = make_float2(tmp4.x * scale.x, tmp4.y * scale.y);
            data[idx4] = make_float2(tmp2.x * scale.x, tmp2.y * scale.y);
        }
    }
} // end kernel_fft_shift
}
