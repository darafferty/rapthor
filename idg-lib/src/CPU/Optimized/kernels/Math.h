#if defined(USE_LOOKUP)
#define PI     float(M_PI)
#define TWO_PI float(2 * M_PI)
#define HLF_PI float(M_PI_2)
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
    const int n,
    float* __restrict__ lookup)
{
    float p = 0;
    float increment = HLF_PI / NR_SAMPLES;
    for (int i = 0; i < NR_SAMPLES+1; i++) {
        lookup[i] = sinf(p);
        p += increment;
    }
}

inline idg::float2 compute_sincos(
    const float* __restrict__ lookup,
    const float x)
{
        float p = x;
        float s1 = 1;
        float s2;

        // Shift p in range [0:2*pi]
        if (p < 0 || p > TWO_PI) {
            p = fmodf(p, TWO_PI);
            p = p < 0 ? p + TWO_PI : p;
        }

        // Shift p in range [0:pi]
        if (p > PI) {
            p = p - PI;
            s1 = -1;
        }

        // Shift p in range [0:0.5*pi]
        if (p > HLF_PI) {
            p = PI - p;
            s2 = -s1;
        } else {
            s2 = s1;
        }

        // Compute indices
        int index1 = (int) ((p * NR_SAMPLES / HLF_PI) + 0.5f);
        int index2 = NR_SAMPLES - index1;

        return {lookup[index2] * s2, lookup[index1] * s1};
}

inline void compute_sincos(
    const float* __restrict__ lookup,
    const float* __restrict__ x,
    const int                 n,
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
