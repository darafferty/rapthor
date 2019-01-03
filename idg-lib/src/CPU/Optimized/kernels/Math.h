#include "common/Math.h"

#define ALIGNMENT 64

#if defined(USE_LOOKUP)
#include "Lookup_01.h"
//#include "Lookup_02.h"
#else
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
    }
    for (int i = 0; i < n; i++) {
            cos[i] = cosf(x[i]);
    }
    #endif
}
#endif

#include "Reduction.h"
