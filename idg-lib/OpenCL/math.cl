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
