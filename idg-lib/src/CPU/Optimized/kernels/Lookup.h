// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "math.h"

#if defined(__x86_64__)
#include <immintrin.h>
#endif

// Floating-point PI values
#define PI float(M_PI)
#define TWO_PI float(2 * M_PI)
#define HLF_PI float(M_PI_2)

// Integer representations of PI
#define TWO_PI_INT 16384
#define PI_INT TWO_PI_INT / 2
#define HLF_PI_INT TWO_PI_INT / 4

// Constants for sine/cosine lookup table
#define NR_SAMPLES TWO_PI_INT

void initialize_lookup();

#if defined(__AVX__) && not defined(__AVX2__)
__m256i _mm256_and_si256(__m256i a, __m256i b);
__m256i _mm256_add_epi32(__m256i a, __m256i b);
#endif

#if defined(__AVX__)
__m256 _mm256_gather_ps(float const* base_addr, __m256i vindex);
#endif

void compute_sincos_avx(unsigned* offset, const unsigned n,
                        const float* __restrict__ x, float* __restrict__ sin,
                        float* __restrict__ cos);

void compute_sincos_altivec(unsigned* offset, const unsigned n,
                            const float* __restrict__ x,
                            float* __restrict__ sin, float* __restrict__ cos);

void compute_sincos_scalar(unsigned* offset, const unsigned n,
                           const float* __restrict__ x, float* __restrict__ sin,
                           float* __restrict__ cos);

void compute_sincos(const unsigned n, const float* __restrict__ x,
                    float* __restrict__ sin, float* __restrict__ cos);
