inline __device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ float2 operator*(float2 a, float b) {
    return make_float2(a.x * b, a.y * b);
}

inline __device__ float2 operator*(float a, float2 b) {
    return make_float2(a * b.x, a * b.y);
}

inline __device__ float2 operator*(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y,
                       a.x * b.y + a.y * b.x);
}

inline __device__ void operator+=(float2 &a, float2 b) {
    a.x += b.x;
    a.y += b.y;
}

inline __device__ void operator*=(float2 &a, float2 b) {
    a.x = a.x * b.x - a.y * b.y;
    a.y = a.x * b.y + a.y * b.x;
}

inline  __device__ void atomicAdd(float2 *a, float2 b) {
    atomicAdd(&a->x, b.x);
    atomicAdd(&a->y, b.y);
}


template <typename T> inline __device__ void Matrix2x2mul(
    T &Cxx, T &Cxy, T &Cyx, T &Cyy,
    T  Axx, T  Axy, T  Ayx, T  Ayy,
    T  Bxx, T  Bxy, T  Byx, T  Byy)
{
    Cxx  = Axx * Bxx + Axy * Byx;
    Cxy  = Axx * Bxy + Axy * Byy;
    Cyx  = Ayx * Bxx + Ayy * Byx;
    Cyy  = Ayx * Bxy + Ayy * Byy;
}
