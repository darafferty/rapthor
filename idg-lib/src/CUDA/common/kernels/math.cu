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
    return make_float2(a.x * b.x - a.y * b.y,
                       a.x * b.y + a.y * b.x);
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

inline __device__ float raw_sin(float a)
{
    float r;
    asm ("sin.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(a));
    return r;
}

inline __device__ float raw_cos(float a)
{
    float r;
    asm ("cos.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(a));
    return r;
}


/*
    Multiply accumulate: a = a + (b * c)
*/
// scalar
inline __device__ void mac(float &a, float b, float c)
{
    asm ("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(a) : "f"(b), "f"(c), "f"(a));
}

// complex
inline __device__ void cmac(float2 &a, float2 b, float2 c)
{
    asm ("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(a.x) : "f"(b.x), "f"(c.x), "f"(a.x));
    asm ("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(a.y) : "f"(b.x), "f"(c.y), "f"(a.y));
    asm ("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(a.x) : "f"(-b.y), "f"(c.y), "f"(a.x));
    asm ("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(a.y) : "f"(b.y), "f"(c.x), "f"(a.y));
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

/*
    Common math functions
*/
#define FUNCTION_ATTRIBUTES __device__
#include "common/Math.h"
