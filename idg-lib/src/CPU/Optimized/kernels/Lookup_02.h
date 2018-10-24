// Constants for sine/cosine lookup table
#define BITS        14
#define QRT_PI_INT  8192 // pow(2, BITS - 1);
#define HLF_PI_INT  (QRT_PI_INT << 1)
#define TWO_PI_INT  (QRT_PI_INT << 3)
#define NR_SAMPLES  (QRT_PI_INT)
#define BIT(A,N)	(((A) >> (N)) & 1)

// Lookup table
#define CREATE_LOOKUP idg::float2 lookup[NR_SAMPLES]; compute_lookup(lookup);

inline void compute_lookup(
    idg::float2* __restrict__ lookup)
{
    for (unsigned i = 0; i < NR_SAMPLES; i++) {
        lookup[i].real = sinf(i * (M_PI_4/QRT_PI_INT));
        lookup[i].imag = cosf(i * (M_PI_4/QRT_PI_INT));
    }
}

inline idg::float2 compute_sincos(
    const idg::float2* __restrict__ lookup,
    const float x)
{
    // Convert p into int
    unsigned index = round(x * (QRT_PI_INT/M_PI_4));

    // Determine quadrant in range [0:2*pi]
    char quadrant = (index >> BITS) & 3;

    // Determine first or second half in quadrant
    char half = BIT(index, BITS - 1);

    // Determine index in range [0:0.25*pi]
    index &= (HLF_PI_INT - 1);
    if (BIT(index, BITS - 1)) {
        index = ~index & (QRT_PI_INT - 1);
    }

    // Lookup sincos
    idg::float2 phasor = lookup[index];

    // Swap real and imag parts
    if (!half ^ BIT(quadrant, 0)) {
        phasor = { phasor.imag, phasor.real };
    }

    // Update real sign
    if (BIT(quadrant, 0) ^ BIT(quadrant, 1)) {
        phasor.real = -phasor.real;
    }

    // Update imag sign
    if (BIT(quadrant, 1) ^ BIT(quadrant, 2)) {
        phasor.imag = -phasor.imag;
    }

    return phasor;
}

inline void compute_sincos(
    const int                       n,
    const float*       __restrict__ x,
    const idg::float2* __restrict__ lookup,
    float*             __restrict__ sin,
    float*             __restrict__ cos)
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
