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
    return tmp > 1.0 ? 1.0 : tmp / (1.0f + sqrtf(1.0f - tmp));
}

/**
 * Calculates n from l, m and the 3 shift parameters.
 * result = 1 - sqrt(1 - lc^2 - mc^2) + pshift
 * with lc = l + lshift, mc = m + mshift
 * @param shift array of size 3 with [lshift, mshift, pshift] parameters
 */
inline float FUNCTION_ATTRIBUTES compute_n(
    float l,
    float m,
    const float* __restrict__ shift)
{
    const float lc = l + shift[0];
    const float mc = m + shift[1];
    const float tmp = (lc * lc) + (mc * mc);
    return tmp > 1.0 ? 1.0 : tmp / (1.0f + sqrtf(1.0f - tmp)) + shift[2];
}

template <typename T> FUNCTION_ATTRIBUTES inline void apply_aterm(
    const T aXX1, const T aXY1,
    const T aYX1, const T aYY1,
    const T aXX2, const T aXY2,
    const T aYX2, const T aYY2,
    T pixels[NR_POLARIZATIONS]
) {
    T pixelsXX = pixels[0];
    T pixelsXY = pixels[1];
    T pixelsYX = pixels[2];
    T pixelsYY = pixels[3];

    // Apply aterm to subgrid: P = A1 * P
    // [ pixels[0], pixels[1];  = [ aXX1, aXY1;  [ pixelsXX, pixelsXY;
    //   pixels[2], pixels[3] ]     aYX1, aYY1 ]   pixelsYX], pixelsYY ] *
    pixels[0]  = (pixelsXX * aXX1);
    pixels[0] += (pixelsYX * aXY1);
    pixels[1]  = (pixelsXY * aXX1);
    pixels[1] += (pixelsYY * aXY1);
    pixels[2]  = (pixelsXX * aYX1);
    pixels[2] += (pixelsYX * aYY1);
    pixels[3]  = (pixelsXY * aYX1);
    pixels[3] += (pixelsYY * aYY1);

    pixelsXX = pixels[0];
    pixelsXY = pixels[1];
    pixelsYX = pixels[2];
    pixelsYY = pixels[3];

    // Apply aterm to subgrid: P = P * A2^H
    //    [ pixels[0], pixels[1];  =   [ pixelsXX, pixelsXY;  *  [ conj(aXX2), conj(aYX2);
    //      pixels[2], pixels[3] ]       pixelsYX, pixelsYY ]      conj(aXY2), conj(aYY2) ]
    pixels[0]  = (pixelsXX * conj(aXX2));
    pixels[0] += (pixelsXY * conj(aXY2));
    pixels[1]  = (pixelsXX * conj(aYX2));
    pixels[1] += (pixelsXY * conj(aYY2));
    pixels[2]  = (pixelsYX * conj(aXX2));
    pixels[2] += (pixelsYY * conj(aXY2));
    pixels[3]  = (pixelsYX * conj(aYX2));
    pixels[3] += (pixelsYY * conj(aYY2));
}

template <typename T> FUNCTION_ATTRIBUTES inline void apply_avg_aterm_correction(
    const T C[16],
    T pixels[NR_POLARIZATIONS])
{
//        [pixel0
//         pixel1
//         pixel2   = vec( [ pixels[0], pixels[1],
//         pixel3]           pixels[2], pixels[3]])

    const T pixel0 = pixels[0];
    const T pixel1 = pixels[2];
    const T pixel2 = pixels[1];
    const T pixel3 = pixels[3];

//     [pixels[0]
//      pixels[1]   = unvec(C * p)
//      pixels[2]
//      pixels[3]]

    pixels[0]  = pixel0*C[ 0] + pixel1*C[ 1] + pixel2*C[ 2] + pixel3*C[ 3];
    pixels[1]  = pixel0*C[ 8] + pixel1*C[ 9] + pixel2*C[10] + pixel3*C[11];
    pixels[2]  = pixel0*C[ 4] + pixel1*C[ 5] + pixel2*C[ 6] + pixel3*C[ 7];
    pixels[3]  = pixel0*C[12] + pixel1*C[13] + pixel2*C[14] + pixel3*C[15];
}

template <typename T> inline FUNCTION_ATTRIBUTES void apply_aterm(
    const T aXX1, const T aXY1, const T aYX1, const T aYY1,
    const T aXX2, const T aXY2, const T aYX2, const T aYY2,
          T &uvXX,      T &uvXY,      T &uvYX,      T &uvYY)
{
    T uv[NR_POLARIZATIONS] = {uvXX, uvXY, uvYX, uvYY};

    apply_aterm(
        aXX1, aXY1, aYX1, aYY1,
        aXX2, aXY2, aYX2, aYY2,
        uv);

    uvXX = uv[0];
    uvXY = uv[1];
    uvYX = uv[2];
    uvYY = uv[3];

}

template <typename T> inline FUNCTION_ATTRIBUTES void apply_avg_aterm_correction(
    const T C[16],
          T &uvXX,      T &uvXY,      T &uvYX,      T &uvYY)
{
    T uv[NR_POLARIZATIONS] = {uvXX, uvXY, uvYX, uvYY};

    apply_avg_aterm_correction(C, uv);

    uvXX = uv[0];
    uvXY = uv[1];
    uvYX = uv[2];
    uvYY = uv[3];

}
