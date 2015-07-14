typedef float2 fcomplex;
typedef float4 fcomplex2;
typedef float8 fcomplex4;

fcomplex cadd(fcomplex a, fcomplex b) {
    return (fcomplex) (a.x + b.x, a.y + b.y);
}

fcomplex cmul(fcomplex a, fcomplex b) {
    return (fcomplex) (a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fcomplex cexp(float ang) {
    return (fcomplex) (native_cos(ang), native_sin(ang));
}


fcomplex conj(fcomplex z) {
    return (fcomplex) (z.x, -z.y);
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

inline void atomicAdd(fcomplex *a, fcomplex b) {
    atomic_add_float(&(*a)->x, b.x);
    atomic_add_float(&(*a)->y, b.x);
}
