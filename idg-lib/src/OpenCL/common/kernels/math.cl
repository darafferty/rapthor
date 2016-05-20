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

inline void apply_aterm(
    const float2 aXX1, const float2 aXY1, const float2 aYX1, const float2 aYY1,
    const float2 aXX2, const float2 aXY2, const float2 aYX2, const float2 aYY2,
          float2 *uvXX,      float2 *uvXY,      float2 *uvYX,      float2 *uvYY
) {
    // Apply aterm to subgrid: P*A1
    // [ uvXX, uvXY;    [ aXX1, aXY1;
    //   uvYX, uvYY ] *   aYX1, aYY1 ]
    float2 pixelsXX = *uvXX;
    float2 pixelsXY = *uvXY;
    float2 pixelsYX = *uvYX;
    float2 pixelsYY = *uvYY;
    *uvXX  = cmul(pixelsXX, aXX1);
    *uvXX += cmul(pixelsXY, aYX1);
    *uvXY  = cmul(pixelsXX, aXY1);
    *uvXY += cmul(pixelsXY, aYY1);
    *uvYX  = cmul(pixelsYX, aXX1);
    *uvYX += cmul(pixelsYY, aYX1);
    *uvYY  = cmul(pixelsYX, aXY1);
    *uvYY += cmul(pixelsYY, aYY1);

    // Apply aterm to subgrid: A2^H*P
    // [ aXX2, aYX1;      [ uvXX, uvXY;
    //   aXY1, aYY2 ]  *    uvYX, uvYY ]
    pixelsXX = *uvXX;
    pixelsXY = *uvXY;
    pixelsYX = *uvYX;
    pixelsYY = *uvYY;
    *uvXX  = cmul(pixelsXX, aXX2);
    *uvXX += cmul(pixelsYX, aYX2);
    *uvXY  = cmul(pixelsXY, aXX2);
    *uvXY += cmul(pixelsYY, aYX2);
    *uvYX  = cmul(pixelsXX, aXY2);
    *uvYX += cmul(pixelsYX, aYY2);
    *uvYY  = cmul(pixelsXY, aXY2);
    *uvYY += cmul(pixelsYY, aYY2);
}
