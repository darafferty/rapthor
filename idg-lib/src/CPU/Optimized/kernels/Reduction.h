#if defined(__x86_64__)
#include <immintrin.h>
#endif

#if defined(__AVX__)
inline float _mm256_horizontal_add(__m256 x) {
    /* x0, x1, x2, x3, x4, x5, x6, x7 */
    __m256 x1 = x;

    /* x0+x1, x2+x3, x0+x1, x2+x3, x4+x5, x6+x7, x4+x5, x6+x7 */
    __m256 x2 = _mm256_hadd_ps(x1, x1);

    /* x0+x1+x2+x3, -, -, -, x4+x5+x6+x7, -, -, - */
    __m256 x3 = _mm256_hadd_ps(x2, x2);

    /* x4+x5+x6+x7, -, -, - */
    __m128 x4 = _mm256_extractf128_ps(x3, 1);

    /* x0+x1+x2+x3, -, -, - */
    __m128 x5 = _mm256_castps256_ps128(x3);

    /* x0+x1+x2+x3+x4+x5+x6+x7, -, -, - */
    __m128 x6 = _mm_add_ss(x4, x5);

    return _mm_cvtss_f32(x6);
}
#endif

// https://bit.ly/2UqZqAp
#if defined(__AVX512F__)
inline float _mm512_horizontal_add(__m512 x) {
    /* x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 */
    __m512 x1 = x;

    /* x8, x9, x10, x11, x12, x13, x14, x15, x0, x1, x2, x3, x4, x5, x6, x7 */
    __m512 x2 = _mm512_shuffle_f32x4(x1,x1,_MM_SHUFFLE(0,0,3,2));

    /* x0+x8, x1+x9, x2+x10, x3+x11, x4+x12, x5+x13, x6+x14, x7+x15, -, -, -, -, -, -, -, - */
    __m512 x3 = _mm512_add_ps(x1, x2);

    /* x4+x12, x5+x13, x6+x14, x7+x15, x0+x8, x1+x9, x2+x10, x3+x11, -, -, -, -, -, -, -, - */
    __m512 x4 = _mm512_shuffle_f32x4(x3,x3,_MM_SHUFFLE(0,0,0,1));

    /* x0+x8+x4+x12, x1+x9+x5+x13, x2+x10+x6+x14, x3+x11+x7+x15, -, -, -, -, -, -, -, -, -, -, -, - */
    __m512 x5 = _mm512_add_ps(x3, x4);

    /* x0+x8+x4+x12, x1+x9+x5+x13, x2+x10+x6+x14, x3+x11+x7+x15 */
    __m128 x6 = _mm512_castps512_ps128(x5);

    /* x0+x8+x4+x12+x1+x9+x5+x13, x2+x10+x6+x14+x3+x11+x7+x15, -, - */
    __m128 x7 = _mm_hadd_ps(x6, x6);

    /* x0+x8+x4+x12+x1+x9+x5+x13+x2+x10+x6+x14+x3+x11+x7+x15, -, - */
    __m128 x8 = _mm_hadd_ps(x7, x7);

    return  _mm_cvtss_f32(x8);
}
#endif
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

    #if defined(__INTEL_COMPILER)
    #pragma omp simd reduction(+:output_xx_real,output_xx_imag, \
                                 output_xy_real,output_xy_imag, \
                                 output_yx_real,output_yx_imag, \
                                 output_yy_real,output_yy_imag)
    #endif
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
#if defined(__AVX2__)
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
        output[0].real += _mm256_horizontal_add(output_xx_r);
        output[1].real += _mm256_horizontal_add(output_xy_r);
        output[2].real += _mm256_horizontal_add(output_yx_r);
        output[3].real += _mm256_horizontal_add(output_yy_r);
        output[0].imag += _mm256_horizontal_add(output_xx_i);
        output[1].imag += _mm256_horizontal_add(output_xy_i);
        output[2].imag += _mm256_horizontal_add(output_yx_i);
        output[3].imag += _mm256_horizontal_add(output_yy_i);
    }

    *offset += vector_length * ((n - *offset) / vector_length);
#endif
} // end compute_reduction_avx2

inline void compute_reduction_avx(
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
#if defined(__AVX__)
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
        output_xx_r = _mm256_add_ps(output_xx_r, _mm256_mul_ps(input_xx, phasor_r));
        output_xx_i = _mm256_add_ps(output_xx_i, _mm256_mul_ps(input_xx, phasor_i));
        output_xy_r = _mm256_add_ps(output_xy_r, _mm256_mul_ps(input_xy, phasor_r));
        output_xy_i = _mm256_add_ps(output_xy_i, _mm256_mul_ps(input_xy, phasor_i));
        output_yx_r = _mm256_add_ps(output_yx_r, _mm256_mul_ps(input_yx, phasor_r));
        output_yx_i = _mm256_add_ps(output_yx_i, _mm256_mul_ps(input_yx, phasor_i));
        output_yy_r = _mm256_add_ps(output_yy_r, _mm256_mul_ps(input_yy, phasor_r));
        output_yy_i = _mm256_add_ps(output_yy_i, _mm256_mul_ps(input_yy, phasor_i));

        // Load imag part of input
        input_xx = _mm256_load_ps(&input_xx_imag[i]);
        input_xy = _mm256_load_ps(&input_xy_imag[i]);
        input_yx = _mm256_load_ps(&input_yx_imag[i]);
        input_yy = _mm256_load_ps(&input_yy_imag[i]);

        // Update output
        output_xx_r = _mm256_sub_ps(output_xx_r, _mm256_mul_ps(input_xx, phasor_i));
        output_xx_i = _mm256_add_ps(output_xx_i, _mm256_mul_ps(input_xx, phasor_r));
        output_xy_r = _mm256_sub_ps(output_xy_r, _mm256_mul_ps(input_xy, phasor_i));
        output_xy_i = _mm256_add_ps(output_xy_i, _mm256_mul_ps(input_xy, phasor_r));
        output_yx_r = _mm256_sub_ps(output_yx_r, _mm256_mul_ps(input_yx, phasor_i));
        output_yx_i = _mm256_add_ps(output_yx_i, _mm256_mul_ps(input_yx, phasor_r));
        output_yy_r = _mm256_sub_ps(output_yy_r, _mm256_mul_ps(input_yy, phasor_i));
        output_yy_i = _mm256_add_ps(output_yy_i, _mm256_mul_ps(input_yy, phasor_r));
    }

    // Reduce all vectors
    if (n - *offset > 0) {
        output[0].real += _mm256_horizontal_add(output_xx_r);
        output[1].real += _mm256_horizontal_add(output_xy_r);
        output[2].real += _mm256_horizontal_add(output_yx_r);
        output[3].real += _mm256_horizontal_add(output_yy_r);
        output[0].imag += _mm256_horizontal_add(output_xx_i);
        output[1].imag += _mm256_horizontal_add(output_xy_i);
        output[2].imag += _mm256_horizontal_add(output_yx_i);
        output[3].imag += _mm256_horizontal_add(output_yy_i);
    }

    *offset += vector_length * ((n - *offset) / vector_length);
#endif
} // end compute_reduction_avx

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
#if defined(__AVX512F__)
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
        output[0].real += _mm512_horizontal_add(output_xx_r);
        output[1].real += _mm512_horizontal_add(output_xy_r);
        output[2].real += _mm512_horizontal_add(output_yx_r);
        output[3].real += _mm512_horizontal_add(output_yy_r);
        output[0].imag += _mm512_horizontal_add(output_xx_i);
        output[1].imag += _mm512_horizontal_add(output_xy_i);
        output[2].imag += _mm512_horizontal_add(output_yx_i);
        output[3].imag += _mm512_horizontal_add(output_yy_i);
    }

    *offset += vector_length * ((n - *offset) / vector_length);
#endif
} // end compute_reduction_avx512

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

    // Vectorized loop, 16-elements, AVX512
    compute_reduction_avx512(
            &offset, n,
            input_xx_real, input_xy_real, input_yx_real, input_yy_real,
            input_xx_imag, input_xy_imag, input_yx_imag, input_yy_imag,
            phasor_real, phasor_imag,
            output);

    // Vectorized loop, 8-elements, AVX2
    compute_reduction_avx2(
            &offset, n,
            input_xx_real, input_xy_real, input_yx_real, input_yy_real,
            input_xx_imag, input_xy_imag, input_yx_imag, input_yy_imag,
            phasor_real, phasor_imag,
            output);

    // Vectorized loop, 8-elements, AVX
    compute_reduction_avx(
            &offset, n,
            input_xx_real, input_xy_real, input_yx_real, input_yy_real,
            input_xx_imag, input_xy_imag, input_yx_imag, input_yy_imag,
            phasor_real, phasor_imag,
            output);

    // Remainder loop, scalar
    compute_reduction_scalar(
            &offset, n,
            input_xx_real, input_xy_real, input_yx_real, input_yy_real,
            input_xx_imag, input_xy_imag, input_yx_imag, input_yy_imag,
            phasor_real, phasor_imag,
            output);
}
