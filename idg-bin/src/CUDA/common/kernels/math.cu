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

template <typename T> inline __device__ void apply_aterm(
    const T aXX1, const T aXY1, const T aYX1, const T aYY1,
    const T aXX2, const T aXY2, const T aYX2, const T aYY2,
          T &uvXX,      T &uvXY,      T &uvYX,      T &uvYY
) {
    // Apply aterm to subgrid: P*A1
    // [ uvXX, uvXY;    [ aXX1, aXY1;
    //   uvYX, uvYY ] *   aYX1, aYY1 ]
    T pixelsXX = uvXX;
    T pixelsXY = uvXY;
    T pixelsYX = uvYX;
    T pixelsYY = uvYY;
    uvXX  = (pixelsXX * aXX1);
    uvXX += (pixelsXY * aYX1);
    uvXY  = (pixelsXX * aXY1);
    uvXY += (pixelsXY * aYY1);
    uvYX  = (pixelsYX * aXX1);
    uvYX += (pixelsYY * aYX1);
    uvYY  = (pixelsYX * aXY1);
    uvYY += (pixelsYY * aYY1);

    // Apply aterm to subgrid: A2^H*P
    // [ aXX2, aYX1;      [ uvXX, uvXY;
    //   aXY1, aYY2 ]  *    uvYX, uvYY ]
    pixelsXX = uvXX;
    pixelsXY = uvXY;
    pixelsYX = uvYX;
    pixelsYY = uvYY;
    uvXX  = (pixelsXX * aXX2);
    uvXX += (pixelsYX * aYX2);
    uvXY  = (pixelsXY * aXX2);
    uvXY += (pixelsYY * aYX2);
    uvYX  = (pixelsXX * aXY2);
    uvYX += (pixelsYX * aYY2);
    uvYY  = (pixelsXY * aXY2);
    uvYY += (pixelsYY * aYY2);
}
