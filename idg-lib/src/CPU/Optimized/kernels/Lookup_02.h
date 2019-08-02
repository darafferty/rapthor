// Constants for sine/cosine lookup table
#define BITS        12
#define QRT_PI_INT  (1 << BITS)
#define ONE_PI_INT  (QRT_PI_INT << 2)
#define NR_SAMPLES  (QRT_PI_INT)
#define BIT(A,N)	(((A) >> (N)) & 1)

// Lookup table
idg::float2 lookup[NR_SAMPLES] __attribute__((aligned(ALIGNMENT)));

inline void initialize_lookup()
{
    for (unsigned i = 0; i < NR_SAMPLES; i++) {
        lookup[i].real = cosf(i * (M_PI_4/QRT_PI_INT));
        lookup[i].imag = sinf(i * (M_PI_4/QRT_PI_INT));
    }
}

inline idg::float2 compute_sincos(
    const float x)
{
    unsigned input  = (unsigned) round(x * (ONE_PI_INT / M_PI));
    unsigned index  = input & (QRT_PI_INT - 1);
    unsigned octant = input >> BITS;

    if (BIT(octant, 0)) {
        index = ~index & (QRT_PI_INT - 1);
    }

    idg::float2 output = lookup[index];

    if (BIT(octant, 0) ^ BIT(octant, 1)) {
        output = { output.imag, output.real };
    }

    if (BIT(octant, 1) ^ BIT(octant, 2)) {
        output.real = -output.real;
    }

    if (BIT(octant, 2)) {
        output.imag = -output.imag;
    }

    return output;
}

inline void compute_sincos(
    const int                       n,
    const float*       __restrict__ x,
    float*             __restrict__ sin,
    float*             __restrict__ cos)
{
    #if defined(__INTEL_COMPILER)
    #pragma vector aligned(x, sin, cos)
    #endif
    for (int i = 0; i < n; i++) {
        idg::float2 phasor = compute_sincos(x[i]);
        cos[i] = phasor.real;
        sin[i] = phasor.imag;
    }
}
