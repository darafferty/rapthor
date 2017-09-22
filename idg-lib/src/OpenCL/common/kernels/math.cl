typedef float2 fcomplex;
typedef float4 fcomplex2;
typedef float8 fcomplex4;

inline float2 cadd(float2 a, float2 b) {
    return (float2) (a.x + b.x, a.y + b.y);
}

inline float2 cmul(float2 a, float2 b) {
    return (float2) (a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline float2 cexp(float ang) {
    return (float2) (native_cos(ang), native_sin(ang));
}


inline float2 conj(float2 z) {
    return (float2) (z.x, -z.y);
}

inline void atomic_add_float(volatile __global float *a, const float b) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *a;
        newVal.floatVal = prevVal.floatVal + b;
    } while (atomic_cmpxchg((volatile __global unsigned int *)a, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

inline void atomicAdd(__global float2 *a, float2 b) {
    __global float *a_ptr = (__global float *) a;
    #if defined(USE_ATOMIC_FETCH_ADD)
    atomic_fetch_add((atomic_int *) a_ptr + 0, b.x);
    atomic_fetch_add((atomic_int *) a_ptr + 1, b.y);
    #else
    atomic_add_float(a_ptr + 0, b.x);
    atomic_add_float(a_ptr + 1, b.y);
    #endif
}

inline float8 cmul8(float8 a, float8 b) {
    return (float8) (
        a.s0 * b.s0 - a.s1 * b.s1,
        a.s0 * b.s1 + a.s1 * b.s0,
        a.s2 * b.s2 - a.s3 * b.s3,
        a.s2 * b.s3 + a.s3 * b.s2,
        a.s4 * b.s4 - a.s5 * b.s5,
        a.s4 * b.s5 + a.s5 * b.s4,
        a.s6 * b.s6 - a.s7 * b.s7,
        a.s6 * b.s7 + a.s7 * b.s6);
}
