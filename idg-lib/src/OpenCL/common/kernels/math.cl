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
    atomic_add_float(a_ptr + 0, b.x);
    atomic_add_float(a_ptr + 1, b.y);
    //atomic_fetch_add((atomic_int *) a_ptr + 0, b.x);
    //atomic_fetch_add((atomic_int *) a_ptr + 1, b.y);
}

inline void Matrix2x2mul(
    float2 *Cxx, float2 *Cxy, float2 *Cyx, float2 *Cyy,
    float2  Axx, float2  Axy, float2  Ayx, float2  Ayy,
    float2  Bxx, float2  Bxy, float2  Byx, float2  Byy)
{
    *Cxx = cadd(cmul(Axx, Bxx), cmul(Axy, Byx));
    *Cxy = cadd(cmul(Axx, Bxy), cmul(Axy, Byy));
    *Cyx = cadd(cmul(Ayx, Bxx), cmul(Ayy, Byx));
    *Cyy = cadd(cmul(Ayx, Bxy), cmul(Ayy, Byy));
}
