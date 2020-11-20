// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/Math.h"

#if defined(USE_LOOKUP)
//#include "Lookup_01.h"
//#include "Lookup_02.h"
#include "Lookup_03.h"
#else
inline void compute_sincos(const int n, const float *x, float *sin,
                           float *cos) {
#if defined(USE_VML)
  vmsSinCos(n, x, sin, cos, VML_PRECISION);
#else
  for (int i = 0; i < n; i++) {
    sin[i] = sinf(x[i]);
  }
  for (int i = 0; i < n; i++) {
    cos[i] = cosf(x[i]);
  }
#endif
}
#endif

#include "Reduction.h"
