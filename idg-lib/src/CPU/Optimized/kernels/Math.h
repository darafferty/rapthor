// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/Math.h"

#if defined(USE_LOOKUP)
#include "Lookup.h"
#elif defined(USE_VML)
#define VML_PRECISION VML_LA
#include <mkl_vml.h>
inline void compute_sincos(const int n, const float* x, float* sin,
                           float* cos) {
  vmsSinCos(n, x, sin, cos, VML_PRECISION);
}
#else
inline void compute_sincos(const int n, const float* x, float* sin,
                           float* cos) {
  for (int i = 0; i < n; i++) {
    sin[i] = sinf(x[i]);
  }
  for (int i = 0; i < n; i++) {
    cos[i] = cosf(x[i]);
  }
}
#endif

#include "Reduction.h"
#include "Extrapolation.h"