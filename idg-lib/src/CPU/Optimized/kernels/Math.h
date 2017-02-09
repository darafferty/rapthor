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
            cos[i] = cosf(x[i]);
        }
    #endif
}


inline void apply_aterm(
    const idg::float2 aXX1, const idg::float2 aXY1,
    const idg::float2 aYX1, const idg::float2 aYY1,
    const idg::float2 aXX2, const idg::float2 aXY2,
    const idg::float2 aYX2, const idg::float2 aYY2,
    idg::float2 pixels[NR_POLARIZATIONS]
) {
    // Apply aterm to subgrid: P*A1
    // [ pixels[0], pixels[1];    [ aXX1, aXY1;
    //   pixels[2], pixels[3] ] *   aYX1, aYY1 ]
    idg::float2 pixelsXX = pixels[0];
    idg::float2 pixelsXY = pixels[1];
    idg::float2 pixelsYX = pixels[2];
    idg::float2 pixelsYY = pixels[3];
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
