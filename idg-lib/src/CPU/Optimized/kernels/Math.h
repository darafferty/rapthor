#include <string.h>
#include <immintrin.h>

#if defined(USE_LOOKUP)
// Floating-point PI values
#define PI     float(M_PI)
#define TWO_PI float(2 * M_PI)
#define HLF_PI float(M_PI_2)

// Integer representations of PI
#define TWO_PI_INT        32768
#define PI_INT            TWO_PI_INT / 2
#define HLF_PI_INT        TWO_PI_INT / 4

// Constants for sine/cosine lookup table
#define TWO_HLF_PI        TWO_PI + HLF_PI
#define TWO_HLF_PI_INT    TWO_PI_INT + HLF_PI_INT
#define NR_SAMPLES        TWO_HLF_PI_INT + 1
#endif

// ALignment
#if defined(__AVX512F__)
#define ALIGNMENT 64
#else
#define ALIGNMENT 32
#endif

inline void compute_sincos(
    const int n,
    const float *x,
    float *sin,
    float *cos
) {
    #if defined(USE_VML)
    vmsSinCos(n, x, sin, cos, VML_PRECISION);
    #else
    for (int i = 0; i < n; i++) {
            sin[i] = sinf(x[i]);
            cos[i] = cosf(x[i]);
    }
    #endif
}

#if defined(USE_LOOKUP)
inline void compute_lookup(
    float* __restrict__ lookup)
{
    for (int i = 0; i < NR_SAMPLES; i++) {
        lookup[i] = sinf(i * (TWO_PI / TWO_PI_INT));
    }
}

inline idg::float2 compute_sincos(
    const float* __restrict__ lookup,
    const float x)
{
        float p = x;

        // Convert to integer pi range [0:NR_SAMPLES]
        int p_int = int(p * (TWO_PI_INT / TWO_PI));

        // Shift p in range [0:2*pi]
        p_int &= (TWO_PI_INT - 1);

        return {
            lookup[p_int+HLF_PI_INT],
            lookup[p_int]};
}

inline void compute_sincos(
    const int                 n,
    const float* __restrict__ x,
    const float* __restrict__ lookup,
    float*       __restrict__ sin,
    float*       __restrict__ cos)
{
    #pragma vector aligned(x, sin, cos)
    for (int i = 0; i < n; i++) {
        idg::float2 phasor = compute_sincos(lookup, x[i]);
        cos[i] = phasor.real;
        sin[i] = phasor.imag;
    }
}
#endif

inline void apply_aterm(
    const idg::float2 aXX1, const idg::float2 aXY1,
    const idg::float2 aYX1, const idg::float2 aYY1,
    const idg::float2 aXX2, const idg::float2 aXY2,
    const idg::float2 aYX2, const idg::float2 aYY2,
    idg::float2 pixels[NR_POLARIZATIONS]
) {
    // Apply aterm to subgrid: P*A1
    // [ pixels[0], pixels[1];    [ aXX1, aXY1;
    //   pixels[2], pixels[3] ] *   aYX1, aYY1 ]
    idg::float2 pixelsXX = pixels[0];
    idg::float2 pixelsXY = pixels[1];
    idg::float2 pixelsYX = pixels[2];
    idg::float2 pixelsYY = pixels[3];
    pixels[0]  = (pixelsXX * aXX1);
    pixels[0] += (pixelsXY * aYX1);
    pixels[1]  = (pixelsXX * aXY1);
    pixels[1] += (pixelsXY * aYY1);
    pixels[2]  = (pixelsYX * aXX1);
    pixels[2] += (pixelsYY * aYX1);
    pixels[3]  = (pixelsYX * aXY1);
    pixels[3] += (pixelsYY * aYY1);

    // Apply aterm to subgrid: A2^H*P
    // [ aXX2, aYX1;      [ pixels[0], pixels[1];
    //   aXY1, aYY2 ]  *    pixels[2], pixels[3] ]
    pixelsXX = pixels[0];
    pixelsXY = pixels[1];
    pixelsYX = pixels[2];
    pixelsYY = pixels[3];
    pixels[0]  = (pixelsXX * aXX2);
    pixels[0] += (pixelsYX * aYX2);
    pixels[1]  = (pixelsXY * aXX2);
    pixels[1] += (pixelsYY * aYX2);
    pixels[2]  = (pixelsXX * aXY2);
    pixels[2] += (pixelsYX * aYY2);
    pixels[3]  = (pixelsXY * aXY2);
    pixels[3] += (pixelsYY * aYY2);
}

// http://bit.ly/2shIfmP
inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1),
                                   _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128,
                                   _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64,
                                   _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

inline void compute_reduction_scalar(
    int *offset,
    const int n,
    const float *input_xx_real,
    const float *input_xy_real,
    const float *input_yx_real,
    const float *input_yy_real,
    const float *input_xx_imag,
    const float *input_xy_imag,
    const float *input_yx_imag,
    const float *input_yy_imag,
    const float *phasor_real,
    const float *phasor_imag,
    idg::float2 output[NR_POLARIZATIONS])
{
    float output_xx_real = 0.0f;
    float output_xy_real = 0.0f;
    float output_yx_real = 0.0f;
    float output_yy_real = 0.0f;
    float output_xx_imag = 0.0f;
    float output_xy_imag = 0.0f;
    float output_yx_imag = 0.0f;
    float output_yy_imag = 0.0f;

    #pragma omp simd reduction(+:output_xx_real,output_xx_imag, \
                                 output_xy_real,output_xy_imag, \
                                 output_yx_real,output_yx_imag, \
                                 output_yy_real,output_yy_imag)
    for (int i = *offset; i < n; i++) {
        float phasor_real_ = phasor_real[i];
        float phasor_imag_ = phasor_imag[i];

        output_xx_real += input_xx_real[i] * phasor_real_;
        output_xx_imag += input_xx_real[i] * phasor_imag_;
        output_xx_real -= input_xx_imag[i] * phasor_imag_;
        output_xx_imag += input_xx_imag[i] * phasor_real_;

        output_xy_real += input_xy_real[i] * phasor_real_;
        output_xy_imag += input_xy_real[i] * phasor_imag_;
        output_xy_real -= input_xy_imag[i] * phasor_imag_;
        output_xy_imag += input_xy_imag[i] * phasor_real_;

        output_yx_real += input_yx_real[i] * phasor_real_;
        output_yx_imag += input_yx_real[i] * phasor_imag_;
        output_yx_real -= input_yx_imag[i] * phasor_imag_;
        output_yx_imag += input_yx_imag[i] * phasor_real_;

        output_yy_real += input_yy_real[i] * phasor_real_;
        output_yy_imag += input_yy_real[i] * phasor_imag_;
        output_yy_real -= input_yy_imag[i] * phasor_imag_;
        output_yy_imag += input_yy_imag[i] * phasor_real_;
    }

    *offset = n;

    // Update output
    output[0] += {output_xx_real, output_xx_imag};
    output[1] += {output_xy_real, output_xy_imag};
    output[2] += {output_yx_real, output_yx_imag};
    output[3] += {output_yy_real, output_yy_imag};
} // end compute_reduction_scalar

#if defined(__AVX2__)
inline void compute_reduction_avx2(
    int *offset,
    const int n,
    const float *input_xx_real,
    const float *input_xy_real,
    const float *input_yx_real,
    const float *input_yy_real,
    const float *input_xx_imag,
    const float *input_xy_imag,
    const float *input_yx_imag,
    const float *input_yy_imag,
    const float *phasor_real,
    const float *phasor_imag,
    idg::float2 output[NR_POLARIZATIONS])
{
    const int vector_length = 8;

    __m256 output_xx_r = _mm256_setzero_ps();
    __m256 output_xy_r = _mm256_setzero_ps();
    __m256 output_yx_r = _mm256_setzero_ps();
    __m256 output_yy_r = _mm256_setzero_ps();
    __m256 output_xx_i = _mm256_setzero_ps();
    __m256 output_xy_i = _mm256_setzero_ps();
    __m256 output_yx_i = _mm256_setzero_ps();
    __m256 output_yy_i = _mm256_setzero_ps();

    for (int i = *offset; i < (n / vector_length) * vector_length; i += vector_length) {
        __m256 input_xx, input_xy, input_yx, input_yy;
        __m256 phasor_r, phasor_i;

        phasor_r  = _mm256_load_ps(&phasor_real[i]);
        phasor_i  = _mm256_load_ps(&phasor_imag[i]);

        // Load real part of input
        input_xx = _mm256_load_ps(&input_xx_real[i]);
        input_xy = _mm256_load_ps(&input_xy_real[i]);
        input_yx = _mm256_load_ps(&input_yx_real[i]);
        input_yy = _mm256_load_ps(&input_yy_real[i]);

        // Update output
        output_xx_r = _mm256_fmadd_ps(input_xx, phasor_r, output_xx_r);
        output_xx_i = _mm256_fmadd_ps(input_xx, phasor_i, output_xx_i);
        output_xy_r = _mm256_fmadd_ps(input_xy, phasor_r, output_xy_r);
        output_xy_i = _mm256_fmadd_ps(input_xy, phasor_i, output_xy_i);
        output_yx_r = _mm256_fmadd_ps(input_yx, phasor_r, output_yx_r);
        output_yx_i = _mm256_fmadd_ps(input_yx, phasor_i, output_yx_i);
        output_yy_r = _mm256_fmadd_ps(input_yy, phasor_r, output_yy_r);
        output_yy_i = _mm256_fmadd_ps(input_yy, phasor_i, output_yy_i);

        // Load imag part of input
        input_xx = _mm256_load_ps(&input_xx_imag[i]);
        input_xy = _mm256_load_ps(&input_xy_imag[i]);
        input_yx = _mm256_load_ps(&input_yx_imag[i]);
        input_yy = _mm256_load_ps(&input_yy_imag[i]);

        // Update output
        output_xx_r = _mm256_fnmadd_ps(input_xx, phasor_i, output_xx_r);
        output_xx_i =  _mm256_fmadd_ps(input_xx, phasor_r, output_xx_i);
        output_xy_r = _mm256_fnmadd_ps(input_xy, phasor_i, output_xy_r);
        output_xy_i =  _mm256_fmadd_ps(input_xy, phasor_r, output_xy_i);
        output_yx_r = _mm256_fnmadd_ps(input_yx, phasor_i, output_yx_r);
        output_yx_i =  _mm256_fmadd_ps(input_yx, phasor_r, output_yx_i);
        output_yy_r = _mm256_fnmadd_ps(input_yy, phasor_i, output_yy_r);
        output_yy_i =  _mm256_fmadd_ps(input_yy, phasor_r, output_yy_i);
    }

    // Reduce all vectors
    if (n - *offset > 0) {
        output[0].real += _mm256_reduce_add_ps(output_xx_r);
        output[1].real += _mm256_reduce_add_ps(output_xy_r);
        output[2].real += _mm256_reduce_add_ps(output_yx_r);
        output[3].real += _mm256_reduce_add_ps(output_yy_r);
        output[0].imag += _mm256_reduce_add_ps(output_xx_i);
        output[1].imag += _mm256_reduce_add_ps(output_xy_i);
        output[2].imag += _mm256_reduce_add_ps(output_yx_i);
        output[3].imag += _mm256_reduce_add_ps(output_yy_i);
    }


    *offset += vector_length * ((n - *offset) / vector_length);
} // end compute_reduction_avx2
#endif

#if defined(__AVX512F__)
inline void compute_reduction_avx512(
    int *offset,
    const int n,
    const float *input_xx_real,
    const float *input_xy_real,
    const float *input_yx_real,
    const float *input_yy_real,
    const float *input_xx_imag,
    const float *input_xy_imag,
    const float *input_yx_imag,
    const float *input_yy_imag,
    const float *phasor_real,
    const float *phasor_imag,
    idg::float2 output[NR_POLARIZATIONS])
{
    const int vector_length = 16;

    __m512 output_xx_r = _mm512_setzero_ps();
    __m512 output_xy_r = _mm512_setzero_ps();
    __m512 output_yx_r = _mm512_setzero_ps();
    __m512 output_yy_r = _mm512_setzero_ps();
    __m512 output_xx_i = _mm512_setzero_ps();
    __m512 output_xy_i = _mm512_setzero_ps();
    __m512 output_yx_i = _mm512_setzero_ps();
    __m512 output_yy_i = _mm512_setzero_ps();

    for (int i = *offset; i < (n / vector_length) * vector_length; i += vector_length) {
        __m512 input_xx, input_xy, input_yx, input_yy;
        __m512 phasor_r, phasor_i;

        phasor_r  = _mm512_load_ps(&phasor_real[i]);
        phasor_i  = _mm512_load_ps(&phasor_imag[i]);

        // Load real part of input
        input_xx = _mm512_load_ps(&input_xx_real[i]);
        input_xy = _mm512_load_ps(&input_xy_real[i]);
        input_yx = _mm512_load_ps(&input_yx_real[i]);
        input_yy = _mm512_load_ps(&input_yy_real[i]);

        // Update output
        output_xx_r = _mm512_fmadd_ps(input_xx, phasor_r, output_xx_r);
        output_xx_i = _mm512_fmadd_ps(input_xx, phasor_i, output_xx_i);
        output_xy_r = _mm512_fmadd_ps(input_xy, phasor_r, output_xy_r);
        output_xy_i = _mm512_fmadd_ps(input_xy, phasor_i, output_xy_i);
        output_yx_r = _mm512_fmadd_ps(input_yx, phasor_r, output_yx_r);
        output_yx_i = _mm512_fmadd_ps(input_yx, phasor_i, output_yx_i);
        output_yy_r = _mm512_fmadd_ps(input_yy, phasor_r, output_yy_r);
        output_yy_i = _mm512_fmadd_ps(input_yy, phasor_i, output_yy_i);

        // Load imag part of input
        input_xx = _mm512_load_ps(&input_xx_imag[i]);
        input_xy = _mm512_load_ps(&input_xy_imag[i]);
        input_yx = _mm512_load_ps(&input_yx_imag[i]);
        input_yy = _mm512_load_ps(&input_yy_imag[i]);

        // Update output
        output_xx_r = _mm512_fnmadd_ps(input_xx, phasor_i, output_xx_r);
        output_xx_i =  _mm512_fmadd_ps(input_xx, phasor_r, output_xx_i);
        output_xy_r = _mm512_fnmadd_ps(input_xy, phasor_i, output_xy_r);
        output_xy_i =  _mm512_fmadd_ps(input_xy, phasor_r, output_xy_i);
        output_yx_r = _mm512_fnmadd_ps(input_yx, phasor_i, output_yx_r);
        output_yx_i =  _mm512_fmadd_ps(input_yx, phasor_r, output_yx_i);
        output_yy_r = _mm512_fnmadd_ps(input_yy, phasor_i, output_yy_r);
        output_yy_i =  _mm512_fmadd_ps(input_yy, phasor_r, output_yy_i);
    }

    // Reduce all vectors
    if (n - *offset > 0) {
        output[0].real += _mm512_reduce_add_ps(output_xx_r);
        output[1].real += _mm512_reduce_add_ps(output_xy_r);
        output[2].real += _mm512_reduce_add_ps(output_yx_r);
        output[3].real += _mm512_reduce_add_ps(output_yy_r);
        output[0].imag += _mm512_reduce_add_ps(output_xx_i);
        output[1].imag += _mm512_reduce_add_ps(output_xy_i);
        output[2].imag += _mm512_reduce_add_ps(output_yx_i);
        output[3].imag += _mm512_reduce_add_ps(output_yy_i);
    }

    *offset += vector_length * ((n - *offset) / vector_length);
} // end compute_reduction_avx512
#endif

inline void compute_reduction(
    const int n,
    const float *input_xx_real,
    const float *input_xy_real,
    const float *input_yx_real,
    const float *input_yy_real,
    const float *input_xx_imag,
    const float *input_xy_imag,
    const float *input_yx_imag,
    const float *input_yy_imag,
    const float *phasor_real,
    const float *phasor_imag,
    idg::float2 output[NR_POLARIZATIONS])
{
    int offset = 0;

    // Initialize output to zero
    memset(output, 0, NR_POLARIZATIONS * sizeof(idg::float2));

    #if defined(__AVX512F__)
    // Vectorized loop, 16-elements, AVX512
    compute_reduction_avx512(
            &offset, n,
            input_xx_real, input_xy_real, input_yx_real, input_yy_real,
            input_xx_imag, input_xy_imag, input_yx_imag, input_yy_imag,
            phasor_real, phasor_imag,
            output);
    #endif

    #if defined(__AVX2__)
    // Vectorized loop, 8-elements, AVX2
    compute_reduction_avx2(
            &offset, n,
            input_xx_real, input_xy_real, input_yx_real, input_yy_real,
            input_xx_imag, input_xy_imag, input_yx_imag, input_yy_imag,
            phasor_real, phasor_imag,
            output);
    #endif

    // Remainder loop, scalar
    compute_reduction_scalar(
            &offset, n,
            input_xx_real, input_xy_real, input_yx_real, input_yy_real,
            input_xx_imag, input_xy_imag, input_yx_imag, input_yy_imag,
            phasor_real, phasor_imag,
            output);
}
