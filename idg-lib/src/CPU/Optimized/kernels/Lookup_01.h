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

inline idg::float2 compute_sincos(
    const float* __restrict__ lookup,
    const float x)
{
    float p = x;

    // Convert to integer pi range [0:NR_SAMPLES]
    unsigned p_int = round(p * (TWO_PI_INT / TWO_PI));

    // Shift p in range [0:2*pi]
    p_int &= (TWO_PI_INT - 1);

    return {
        lookup[(p_int+HLF_PI_INT) & (TWO_PI_INT - 1)],
        lookup[p_int]};
}

inline void compute_sincos(
    const int                 n,
    const float* __restrict__ x,
    const float* __restrict__ lookup,
    float*       __restrict__ sin,
    float*       __restrict__ cos)
{
    #if defined(__INTEL_COMPILER)
    #pragma vector aligned(x, sin, cos)
    #endif
    for (int i = 0; i < n; i++) {
        idg::float2 phasor = compute_sincos(lookup, x[i]);
        cos[i] = phasor.real;
        sin[i] = phasor.imag;
    }
}
