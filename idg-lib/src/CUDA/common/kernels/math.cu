// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cuComplex.h>
inline __device__ float2 conj(float2 a) {
    return cuConjf(a);
}

inline __device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ float2 operator*(float2 a, float b) {
    return make_float2(a.x * b, a.y * b);
}

inline __device__ float2 operator*(float a, float2 b) {
    return make_float2(a * b.x, a * b.y);
}

inline __device__ float2 operator*(float2 a, float2 b) {
    return make_float2(fma(a.x, b.x, -a.y * b.y),
                       fma(a.x, b.y, a.y * b.x));
}

inline __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __device__ float4 operator*(float a, float4 b) {
    return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

inline __device__ void operator+=(float2 &a, float2 b) {
    a.x += b.x;
    a.y += b.y;
}

inline __device__ void operator+=(double2 &a, double2 b) {
    a.x += b.x;
    a.y += b.y;
}

inline __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(
			address_as_ull, assumed,
			__double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

inline  __device__ void atomicAdd(float2 &a, float2 b) {
    atomicAdd(&(a.x), b.x);
    atomicAdd(&(a.y), b.y);
}

inline  __device__ void atomicAdd(double2 &a, double2 b) {
    atomicAdd(&(a.x), b.x);
    atomicAdd(&(a.y), b.y);
}

inline __device__ void cmac(float2 &a, float2 b, float2 c)
{
    a.x = fma(b.x, c.x, a.x);
    a.y = fma(b.x, c.y, a.y);
    a.x = fma(-b.y, c.y, a.x);
    a.y = fma(b.y, c.x, a.y);
}

template <typename T>
inline __device__ void apply_avg_aterm_correction_(
    const T C[16], T pixels[4]) {

  const T p[4] = {pixels[0], pixels[2], pixels[1], pixels[3]};

  #pragma unroll 1
  for (int i = 0; i < 4; i++)
  {
    int offset = 0;
    switch (i) {
        case 1: offset = 8; break;
        case 2: offset = 4; break;
        case 3: offset = 12; break;
    }
    pixels[i]  = p[0] * C[offset + 0];
    pixels[i] += p[1] * C[offset + 1];
    pixels[i] += p[2] * C[offset + 2];
    pixels[i] += p[3] * C[offset + 3];
  }
}

inline __device__ void compute_reduction(
    int n, int m, int nr_polarizations,
    const float* wavenumbers,
    const float* phase_index,
    const float* phase_offset,
    const float4* input_ptr1,
    const float4* input_ptr2,
    float2* output,
    int input_stride,    // gridder: 1, degridder: 0
    int output_index)  { // gridder: 1, degridder: 0
    for (int i = 0; i < n; i++) {
        const float4 a = input_ptr1[i * input_stride];
        const float4 b = input_ptr2[i * input_stride];

        for (int j = 0; j < m; j++) {
            float phase = fma(wavenumbers[i], phase_index[j], phase_offset[j]);
            float2 phasor;
            __sincosf(phase, &phasor.y, &phasor.x);

            int idx = 4 * (output_index ? j : i);
            cmac(output[idx + 0], phasor, make_float2(a.x, a.y));
            if (nr_polarizations == 4) {
                cmac(output[idx + 1], phasor, make_float2(a.z, a.w));
                cmac(output[idx + 2], phasor, make_float2(b.x, b.y));
            }
            cmac(output[idx + 3], phasor, make_float2(b.z, b.w));
        }
    }
}

/*
    Common math functions
*/
#define FUNCTION_ATTRIBUTES __device__
#include "common/Math.h"
