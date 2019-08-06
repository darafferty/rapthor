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
float lookup[NR_SAMPLES] __attribute__((aligned(ALIGNMENT)));

inline void initialize_lookup()
{
    for (unsigned i = 0; i < NR_SAMPLES; i++) {
        lookup[i] = sinf(i * (TWO_PI / TWO_PI_INT));
    }
}

#if not defined(__AVX2__)
__m256i _mm256_and_si256(__m256i a, __m256i b)
{
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

__m256i _mm256_add_epi32(__m256i a, __m256i b)
{
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

__m256 _mm256_gather_ps(float const* base_addr, __m256i vindex)
{
	float dst[8];
	int idx[8];
	_mm256_store_si256((__m256i *) idx, vindex);
	for (unsigned i = 0; i < 8; i++) {
		dst[i] = base_addr[idx[i]];
	}
	return _mm256_load_ps(dst);
}

inline void compute_sincos_avx(
    unsigned*                 offset,
    const unsigned            n,
    const float* __restrict__ x,
    float*       __restrict__ sin,
    float*       __restrict__ cos)
{
#if defined(__AVX__)
    const unsigned vector_length = 8;

	__m256 two_pi_f      = _mm256_set1_ps(TWO_PI);
	__m256 two_pi_inv_f  = _mm256_set1_ps(1 / TWO_PI);
	__m256 two_pi_int_f  = _mm256_set1_ps(TWO_PI_INT);
	__m256  scale_f      = _mm256_mul_ps(two_pi_int_f, two_pi_inv_f);
	__m256i hlf_pi_int_i = _mm256_set1_epi32(HLF_PI_INT);
	__m256i mask_i       = _mm256_set1_epi32(TWO_PI_INT - 1);

    for (unsigned i = *offset; i < (n / vector_length) * vector_length; i += vector_length) {
	    __m256  f0 = _mm256_load_ps(&x[i]);				 // load input
	    __m256  f1 = _mm256_mul_ps(f0, two_pi_inv_f);    // divide input by 2 * pi
	    __m256i i0 = _mm256_cvtps_epi32(f1);             // get integer part
	    __m256  f2 = _mm256_cvtepi32_ps(i0);             // convert to float
        #if defined(__AVX2__)
        __m256  f4 = _mm256_fnmadd_ps(f2, two_pi_f, f0); // normalize input
        #else
	    __m256  f3 = _mm256_mul_ps(f2, two_pi_f);        // get multiple of 2 * pi
	    __m256  f4 = _mm256_sub_ps(f0, f3);              // normalize input
        #endif
	    __m256  f5 = _mm256_mul_ps(f4, scale_f);         // apply scale
	    __m256i i1 = _mm256_cvtps_epi32(f5);             // convert to int
	    __m256i i2 = _mm256_add_epi32(i1, hlf_pi_int_i); // shift by 0.5 * pi
	    __m256i i3 = _mm256_and_si256(i2, mask_i);       // apply mask, first index
	    __m256i i4 = _mm256_and_si256(i1, mask_i);       // apply mask, second index
	    #if defined(__AVX2__)
	    __m256  f6 = _mm256_i32gather_ps(lookup, i3, 4); // lookup cosine
	    __m256  f7 = _mm256_i32gather_ps(lookup, i4, 4); // lookup sine
	    #else
	    __m256  f6 = _mm256_gather_ps(lookup, i3);       // lookup cosine
	    __m256  f7 = _mm256_gather_ps(lookup, i4);       // lookup sine
	    #endif
	    _mm256_store_ps(&cos[i], f6);                    // store output
	    _mm256_store_ps(&sin[i], f7);                    // store output
    }

    *offset += vector_length * ((n - *offset) / vector_length);
#endif
}

inline void compute_sincos_scalar(
    unsigned*                 offset,
    const unsigned            n,
    const float* __restrict__ x,
    float*       __restrict__ sin,
    float*       __restrict__ cos)
{
	const float two_pi_f     = TWO_PI;
	const float two_pi_inv_f = 1 / TWO_PI;
	const float two_pi_int_f = TWO_PI_INT;
	const float scale_f      = two_pi_int_f * two_pi_inv_f;
	const int hlf_pi_int_i   = HLF_PI_INT;
	const int mask_i         = TWO_PI_INT - 1;

    for (unsigned i = *offset; i < n; i++) {
        float f0 = x[i];		      // load input
		float f1 = f0 * two_pi_inv_f; // divide input by 2 * pi
		int   i0 = (int) f1;          // get integer part
		float f2 = (float) i0;		  // convert to float
		float f3 = f2 * two_pi_f;	  // get multiple of 2 * pi
		float f4 = f0 - f3;			  // normalize input
		float f5 = f4 * scale_f;	  // apply scale
        int   i1 = (int) round(f5);	  // convert to int
		int   i2 = i1 + hlf_pi_int_i; // shift by 0.5 * pi
		int   i3 = i2 & mask_i;       // apply mask, first index
		int   i4 = i1 & mask_i;       // apply mask, second index
		float f6 = lookup[i3];	      // lookup cosine
		float f7 = lookup[i4];	      // lookup sine
		cos[i] = f6;	              // store output
		sin[i] = f7;	              // store output
    }

    *offset = n;
}

inline void compute_sincos(
    const unsigned            n,
    const float* __restrict__ x,
    float*       __restrict__ sin,
    float*       __restrict__ cos)
{
    unsigned offset = 0;
    compute_sincos_avx(&offset, n, x, sin, cos);
    compute_sincos_scalar(&offset, n, x, sin, cos);
}
