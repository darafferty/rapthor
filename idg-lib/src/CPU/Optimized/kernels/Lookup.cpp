// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Lookup.h"

#if defined(__PPC__)
#include "powerveclib/powerveclib.h"
#endif

// Lookup table
#define ALIGNMENT 64
float lookup[NR_SAMPLES] __attribute__((aligned(ALIGNMENT)));

void initialize_lookup() {
  for (unsigned i = 0; i < NR_SAMPLES; i++) {
    lookup[i] = sinf(i * (TWO_PI / TWO_PI_INT));
  }
}

#if defined(__AVX__) && not defined(__AVX2__)
__m256i _mm256_and_si256(__m256i a, __m256i b) {
  __m128i ah = _mm256_extractf128_si256(a, 0);
  __m128i al = _mm256_extractf128_si256(a, 1);
  __m128i bh = _mm256_extractf128_si256(b, 0);
  __m128i bl = _mm256_extractf128_si256(b, 1);
  __m128i ch = _mm_and_si128(ah, bh);
  __m128i cl = _mm_and_si128(al, bl);
#if __GNUC__ > 7 || defined(__INTEL_COMPILER)
  return _mm256_set_m128i(ch, cl);
#else
  return _mm256_insertf128_si256(_mm256_castsi128_si256(ch), cl, 1);
#endif
}

__m256i _mm256_add_epi32(__m256i a, __m256i b) {
  __m128i ah = _mm256_extractf128_si256(a, 0);
  __m128i al = _mm256_extractf128_si256(a, 1);
  __m128i bh = _mm256_extractf128_si256(b, 0);
  __m128i bl = _mm256_extractf128_si256(b, 1);
  __m128i ch = _mm_add_epi32(ah, bh);
  __m128i cl = _mm_add_epi32(al, bl);
#if __GNUC__ > 7 || defined(__INTEL_COMPILER)
  return _mm256_set_m128i(ch, cl);
#else
  return _mm256_insertf128_si256(_mm256_castsi128_si256(ch), cl, 1);
#endif
}
#endif

#if defined(__AVX__)
__m256 _mm256_gather_ps(float const* base_addr, __m256i vindex) {
  float dst[8];
  int idx[8];
  _mm256_store_si256((__m256i*)idx, vindex);
  for (unsigned i = 0; i < 8; i++) {
    dst[i] = base_addr[idx[i]];
  }
  return _mm256_load_ps(dst);
}
#endif

void compute_sincos_avx(unsigned* offset, const unsigned n,
                        const float* __restrict__ x, float* __restrict__ sin,
                        float* __restrict__ cos) {
#if defined(__AVX__)
  const unsigned vector_length = 8;

  for (unsigned i = *offset; i < (n / vector_length) * vector_length;
       i += vector_length) {
    __m256 f0 = _mm256_load_ps(&x[i]);                // input
    __m256 f1 = _mm256_set1_ps(TWO_PI_INT / TWO_PI);  // compute scale
    __m256 f2 = _mm256_mul_ps(f0, f1);                // apply scale
    __m256i u0 = _mm256_set1_epi32(HLF_PI_INT);       // constant 0.5 * pi
    __m256i u1 = _mm256_set1_epi32(TWO_PI_INT - 1);   // mask 2 * pi
    __m256i u2 = _mm256_cvtps_epi32(f2);              // round float to int
    __m256i u3 = _mm256_add_epi32(u2, u0);            // add 0.5 * pi
    __m256i u4 =
        _mm256_and_si256(u1, u3);  // apply mask of 2 * pi, second index
    __m256i u5 = _mm256_and_si256(u1, u2);  // apply mask of 2 * pi, first index
#if defined(__AVX2__)
    __m256 f3 = _mm256_i32gather_ps(lookup, u4, 4);  // perform lookup of real
    __m256 f4 = _mm256_i32gather_ps(lookup, u5, 4);  // perform lookup of imag
#else
    __m256 f3 = _mm256_gather_ps(lookup, u4);  // perform lookup of real
    __m256 f4 = _mm256_gather_ps(lookup, u5);  // perform lookup of imag
#endif
    _mm256_store_ps(&cos[i], f3);  // store output
    _mm256_store_ps(&sin[i], f4);  // store output
  }

  *offset += vector_length * ((n - *offset) / vector_length);
#endif
}

void compute_sincos_altivec(unsigned* offset, const unsigned n,
                            const float* __restrict__ x,
                            float* __restrict__ sin, float* __restrict__ cos) {
#if defined(__PPC__)
  const unsigned vector_length = 4;

  for (unsigned i = *offset; i < (n / vector_length) * vector_length;
       i += vector_length) {
    __m128 f0 = vec_load4sp(&x[i]);             // input
    __m128i u0 = vec_splat4sw(HLF_PI_INT);      // constant 0.5 * pi
    __m128i u1 = vec_splat4sw(TWO_PI_INT - 1);  // mask 2 * pi
    __m128i u2 = vec_convert4spto4sw(f0);       // round float to int
    __m128i u3 = vec_add(u2, u0);               // add 0.5 * pi
    __m128i u4 = vec_bitand1q(u1, u3);  // apply mask of 2 * pi, second index
    __m128i u5 = vec_bitand1q(u1, u2);  // apply mask of 2 * pi, first index
    __m128 f3 = vec_gather4sp(lookup, u4);  // perform lookup of real
    __m128 f4 = vec_gather4sp(lookup, u5);  // perform lookup of imag
    vec_store4sp(&cos[i], f3);
    vec_store4sp(&sin[i], f4);
  }

  *offset += vector_length * ((n - *offset) / vector_length);
#endif
}

void compute_sincos_scalar(unsigned* offset, const unsigned n,
                           const float* __restrict__ x, float* __restrict__ sin,
                           float* __restrict__ cos) {
  for (unsigned i = *offset; i < n; i++) {
    unsigned index = round(x[i] * (TWO_PI_INT / TWO_PI));
    index &= (TWO_PI_INT - 1);
    cos[i] = lookup[(index + HLF_PI_INT) & (TWO_PI_INT - 1)];
    sin[i] = lookup[index];
  }

  *offset = n;
}

void compute_sincos(const unsigned n, const float* __restrict__ x,
                    float* __restrict__ sin, float* __restrict__ cos) {
  unsigned offset = 0;

  compute_sincos_altivec(&offset, n, x, sin, cos);
  compute_sincos_avx(&offset, n, x, sin, cos);
  compute_sincos_scalar(&offset, n, x, sin, cos);
}