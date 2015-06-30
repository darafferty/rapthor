float2 cmul(float2 a, float2 b) {
    return (float2) (a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}


float2 cexp(float ang) {
    return (float2) (native_cos(ang), native_sin(ang));
}


float2 conj(float2 z) {
    return (float2) (z.x, -z.y);
}
