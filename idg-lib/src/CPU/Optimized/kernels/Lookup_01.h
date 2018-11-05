#if defined(__x86_64__)
#include <immintrin.h>
#endif

// Floating-point PI values
#define PI     float(M_PI)
#define TWO_PI float(2 * M_PI)
#define HLF_PI float(M_PI_2)

// Integer representations of PI
#define TWO_PI_INT        16384
#define PI_INT            TWO_PI_INT / 2
#define HLF_PI_INT        TWO_PI_INT / 4

// Constants for sine/cosine lookup table
#define NR_SAMPLES        TWO_PI_INT

// Lookup table
#define CREATE_LOOKUP float lookup[NR_SAMPLES]; compute_lookup(lookup);

inline void compute_lookup(
    float* __restrict__ lookup)
{
    for (unsigned i = 0; i < NR_SAMPLES; i++) {
        lookup[i] = sinf(i * (TWO_PI / TWO_PI_INT));
    }
}

inline void compute_sincos_avx2(
    unsigned*                 offset,
    const unsigned            n,
    const float* __restrict__ x,
    const float* __restrict__ lookup,
    float*       __restrict__ sin,
    float*       __restrict__ cos)
{
#if defined(__AVX2__)
    const unsigned vector_length = 8;

    for (unsigned i = *offset; i < (n / vector_length) * vector_length; i += vector_length) {
        __m256  f0 = _mm256_load_ps(&x[i]);              // input
        __m256  f1 = _mm256_set1_ps(TWO_PI_INT/TWO_PI);  // compute scale
        __m256  f2 = _mm256_mul_ps(f0, f1);              // apply scale
        __m256i u0 = _mm256_set1_epi32(HLF_PI_INT);      // constant 0.5 * pi
        __m256i u1 = _mm256_set1_epi32(TWO_PI_INT - 1);  // mask 2 * pi
        __m256i u2 = _mm256_cvtps_epi32(f2);             // round float to int
        __m256i u3 = _mm256_add_epi32(u2, u0);           // add 0.5 * pi
        __m256i u4 = _mm256_and_si256(u1, u3);           // apply mask of 2 * pi, second index
        __m256i u5 = _mm256_and_si256(u1, u2);           // apply mask of 2 * pi, first index
        __m256  f3 = _mm256_i32gather_ps(lookup, u4, 4); // perform lookup of real
        __m256  f4 = _mm256_i32gather_ps(lookup, u5, 4); // perform lookup of imag
        _mm256_store_ps(&cos[i], f3);
        _mm256_store_ps(&sin[i], f4);
    }

    *offset += vector_length * ((n - *offset) / vector_length);
#endif
}

inline void compute_sincos_scalar(
    unsigned*                 offset,
    const unsigned            n,
    const float* __restrict__ x,
    const float* __restrict__ lookup,
    float*       __restrict__ sin,
    float*       __restrict__ cos)
{
    for (unsigned i = *offset; i < n; i++) {
        unsigned index = round(x[i] * (TWO_PI_INT / TWO_PI));
        index &= (TWO_PI_INT - 1);
        cos[i] = lookup[(index+HLF_PI_INT) & (TWO_PI_INT - 1)];
        sin[i] = lookup[index];
    }

    *offset = n;
}

inline void compute_sincos(
    const unsigned            n,
    const float* __restrict__ x,
    const float* __restrict__ lookup,
    float*       __restrict__ sin,
    float*       __restrict__ cos)
{
    unsigned offset = 0;
    compute_sincos_avx2(&offset, n, x, lookup, sin, cos);
    compute_sincos_scalar(&offset, n, x, lookup, sin, cos);
}
