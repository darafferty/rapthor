#ifndef FUNCTION_ATTRIBUTES
#define FUNCTION_ATTRIBUTES
#endif

inline float FUNCTION_ATTRIBUTES compute_l(
    int x,
    int subgrid_size,
    float image_size)
{
    return (x+0.5-(subgrid_size/2)) * image_size/subgrid_size;
}

inline float FUNCTION_ATTRIBUTES compute_m(
    int y,
    int subgrid_size,
    float image_size)
{
    return compute_l(y, subgrid_size, image_size);
}

inline float FUNCTION_ATTRIBUTES compute_n(
    float l,
    float m)
{
    // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
    // accurately for small values of l and m
    const float tmp = (l * l) + (m * m);
    return tmp / (1.0f + sqrtf(1.0f - tmp));
}

template <typename T> FUNCTION_ATTRIBUTES inline void apply_aterm(
    const T aXX1, const T aXY1,
    const T aYX1, const T aYY1,
    const T aXX2, const T aXY2,
    const T aYX2, const T aYY2,
    T pixels[NR_POLARIZATIONS]
) {
    // Apply aterm to subgrid: P*A1
    // [ pixels[0], pixels[1];    [ aXX1, aXY1;
    //   pixels[2], pixels[3] ] *   aYX1, aYY1 ]
    T pixelsXX = pixels[0];
    T pixelsXY = pixels[1];
    T pixelsYX = pixels[2];
    T pixelsYY = pixels[3];
    pixels[0]  = (pixelsXX * aXX1);
    pixels[0] += (pixelsXY * aYX1);
    pixels[1]  = (pixelsXX * aXY1);
    pixels[1] += (pixelsXY * aYY1);
    pixels[2]  = (pixelsYX * aXX1);
    pixels[2] += (pixelsYY * aYX1);
    pixels[3]  = (pixelsYX * aXY1);
    pixels[3] += (pixelsYY * aYY1);

    // Apply aterm to subgrid: A2^H*P
    // [ aXX2, aYX1;      [ pixels[0], pixels[1];
    //   aXY1, aYY2 ]  *    pixels[2], pixels[3] ]
    pixelsXX = pixels[0];
    pixelsXY = pixels[1];
    pixelsYX = pixels[2];
    pixelsYY = pixels[3];
    pixels[0]  = (pixelsXX * aXX2);
    pixels[0] += (pixelsYX * aYX2);
    pixels[1]  = (pixelsXY * aXX2);
    pixels[1] += (pixelsYY * aYX2);
    pixels[2]  = (pixelsXX * aXY2);
    pixels[2] += (pixelsYX * aYY2);
    pixels[3]  = (pixelsXY * aXY2);
    pixels[3] += (pixelsYY * aYY2);
}

template <typename T> inline FUNCTION_ATTRIBUTES void apply_aterm(
    const T aXX1, const T aXY1, const T aYX1, const T aYY1,
    const T aXX2, const T aXY2, const T aYX2, const T aYY2,
          T &uvXX,      T &uvXY,      T &uvYX,      T &uvYY
) {
    T uv[NR_POLARIZATIONS] = {uvXX, uvXY, uvYX, uvYY};

    return apply_aterm(
            aXX1, aXY1, aYX1, aYY1,
            aXX2, aXY2, aYX2, aYY2,
            uv);
}
