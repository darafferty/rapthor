// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_MATH_H_
#define IDG_MATH_H_

#ifndef FUNCTION_ATTRIBUTES
#define FUNCTION_ATTRIBUTES
#endif

inline float FUNCTION_ATTRIBUTES compute_l(int x, int subgrid_size,
                                           float image_size) {
  return (x + 0.5 - (subgrid_size / 2)) * image_size / subgrid_size;
}

inline float FUNCTION_ATTRIBUTES compute_m(int y, int subgrid_size,
                                           float image_size) {
  return compute_l(y, subgrid_size, image_size);
}

inline float FUNCTION_ATTRIBUTES compute_n(float l, float m) {
  // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
  // accurately for small values of l and m
  const float tmp = (l * l) + (m * m);
  return tmp > 1.0 ? 1.0 : tmp / (1.0f + sqrtf(1.0f - tmp));
}

/**
 * Calculates n from l, m and the 3 shift parameters.
 * result = 1 - sqrt(1 - lc^2 - mc^2) + pshift
 * with lc = l - lshift, mc = m - mshift
 * @param shift array of size 2 with [lshift, mshift] parameters
 *        lshift is positive if the rA is larger than the rA at the center.
 *        mshift is positive if the dec is larger than the dec at the center.
 */
inline float FUNCTION_ATTRIBUTES compute_n(float l, float m,
                                           const float* __restrict__ shift) {
  return compute_n(l - shift[0], m - shift[1]);
}

#define NR_CORRELATIONS_ATERM 4

template <typename T>
FUNCTION_ATTRIBUTES inline void apply_avg_aterm_correction(
    const T C[16], T pixels[NR_CORRELATIONS_ATERM]) {
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

  pixels[0] = pixel0 * C[0] + pixel1 * C[1] + pixel2 * C[2] + pixel3 * C[3];
  pixels[1] = pixel0 * C[8] + pixel1 * C[9] + pixel2 * C[10] + pixel3 * C[11];
  pixels[2] = pixel0 * C[4] + pixel1 * C[5] + pixel2 * C[6] + pixel3 * C[7];
  pixels[3] = pixel0 * C[12] + pixel1 * C[13] + pixel2 * C[14] + pixel3 * C[15];
}

template <typename T>
inline FUNCTION_ATTRIBUTES void apply_avg_aterm_correction(const T C[16],
                                                           T& uvXX, T& uvXY,
                                                           T& uvYX, T& uvYY) {
  T uv[NR_CORRELATIONS_ATERM] = {uvXX, uvXY, uvYX, uvYY};

  apply_avg_aterm_correction(C, uv);

  uvXX = uv[0];
  uvXY = uv[1];
  uvYX = uv[2];
  uvYY = uv[3];
}

template <typename T>
inline FUNCTION_ATTRIBUTES void matmul(const T* a, const T* b, T* c) {
  c[0] = a[0] * b[0];
  c[1] = a[0] * b[1];
  c[2] = a[2] * b[0];
  c[3] = a[2] * b[1];
  c[0] += a[1] * b[2];
  c[1] += a[1] * b[3];
  c[2] += a[3] * b[2];
  c[3] += a[3] * b[3];
}

template <typename T>
inline FUNCTION_ATTRIBUTES void conjugate(const T* a, T* b) {
  float s[8] = {1, -1, 1, -1, 1, -1, 1, -1};
  float* a_ptr = (float*)a;
  float* b_ptr = (float*)b;

  for (unsigned i = 0; i < 8; i++) {
    b_ptr[i] = s[i] * a_ptr[i];
  }
}

FUNCTION_ATTRIBUTES inline int next_composite(int n) {
  n += (n & 1);
  while (true) {
    int nn = n;
    while ((nn % 2) == 0) nn /= 2;
    while ((nn % 3) == 0) nn /= 3;
    while ((nn % 5) == 0) nn /= 5;
    if (nn == 1) return n;
    n += 2;
  }
}

template <typename T>
inline FUNCTION_ATTRIBUTES void transpose(const T* a, T* b) {
  b[0] = a[0];
  b[1] = a[2];
  b[2] = a[1];
  b[3] = a[3];
}

template <typename T>
inline FUNCTION_ATTRIBUTES void hermitian(const T* a, T* b) {
  T temp[4];
  conjugate(a, temp);
  transpose(temp, b);
}

template <typename T>
inline FUNCTION_ATTRIBUTES void apply_aterm_gridder(T* pixels, const T* aterm1,
                                                    const T* aterm2) {
  // Aterm 1 hermitian
  T aterm1_h[4];
  hermitian(aterm1, aterm1_h);

  // Apply aterm: P = A1^H * P
  T temp[4];
  matmul(aterm1_h, pixels, temp);

  // Apply aterm: P = P * A2
  matmul(temp, aterm2, pixels);
}

template <typename T>
inline FUNCTION_ATTRIBUTES void apply_aterm_degridder(T* pixels,
                                                      const T* aterm1,
                                                      const T* aterm2) {
  // Apply aterm: P = A1 * P
  T temp[4];
  matmul(aterm1, pixels, temp);

  // Aterm 2 hermitian
  T aterm2_h[4];
  hermitian(aterm2, aterm2_h);

  // Apply aterm: P = P * A2^H
  matmul(temp, aterm2_h, pixels);
}

#endif  // IDG_MATH_H_