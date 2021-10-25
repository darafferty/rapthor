// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_TYPES_H_
#define IDG_TYPES_H_

#include <iostream>
#include <ostream>
#include <complex>
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>

#ifndef FUNCTION_ATTRIBUTES
#define FUNCTION_ATTRIBUTES
#endif

namespace idg {

/* Structures */
#include "KernelTypes.h"

template <class T>
struct Matrix2x2 {
  T xx;
  T xy;
  T yx;
  T yy;
};

/* Debugging */
template <typename T>
inline bool isnan(T& value) {
  return (std::isnan(value));
}

template <typename T>
inline bool isnan(std::complex<T>& value) {
  return (std::isnan(value.real()) || std::isnan(value.imag()));
}

template <typename T>
inline bool isnan(Matrix2x2<std::complex<T>>& m) {
  return (isnan(m.xx) || isnan(m.xy) || isnan(m.yx) || isnan(m.yy));
}

template <typename T>
inline bool isnan(UVW<T>& uvw) {
  return (std::isnan(uvw.u) || std::isnan(uvw.v) || std::isnan(uvw.w));
}

template <typename T>
inline bool isfinite(T& value) {
  return (std::isfinite(value));
}

template <typename T>
inline bool isfinite(std::complex<T>& value) {
  return (std::isfinite(value.real()) && std::isfinite(value.imag()));
}

template <typename T>
inline bool isfinite(Matrix2x2<std::complex<T>>& m) {
  return (isfinite(m.xx) && isfinite(m.xy) && isfinite(m.yx) && isfinite(m.yy));
}

template <typename T>
inline bool isfinite(UVW<T>& uvw) {
  return (std::isfinite(uvw.u) && std::isfinite(uvw.v) && std::isfinite(uvw.w));
}

/* Output */
std::ostream& operator<<(std::ostream& os, const Baseline& b);
std::ostream& operator<<(std::ostream& os, const Coordinate& c);
std::ostream& operator<<(std::ostream& os, const Metadata& m);

template <class T>
std::ostream& operator<<(std::ostream& out,
                         const Matrix2x2<std::complex<T>>& m) {
  out << "(" << m.xx << "," << m.xy << "," << m.yx << "," << m.yy << ")";
  return out;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const UVW<T>& uvw);

}  // end namespace idg

#endif
