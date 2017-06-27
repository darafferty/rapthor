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
