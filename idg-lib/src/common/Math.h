inline float compute_l(
    int x,
    int subgrid_size,
    float image_size)
{
    return (x+0.5-(subgrid_size/2)) * image_size/subgrid_size;
}

inline float compute_m(
    int y,
    int subgrid_size,
    float image_size)
{
    return compute_l(y, subgrid_size, image_size);
}

inline float compute_n(
    float l,
    float m)
{
    // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
    // accurately for small values of l and m
    const float tmp = (l * l) + (m * m);
    return tmp / (1.0f + sqrtf(1.0f - tmp));
}
