// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Types.h"

using namespace std;

namespace idg {

ostream& operator<<(ostream& out, Baseline& b) {
  out << "(" << b.station1 << "," << b.station2 << ")";
  return out;
}

ostream& operator<<(ostream& out, Coordinate& c) {
  out << "(" << c.x << "," << c.y << ")";
  return out;
}

ostream& operator<<(ostream& out, Metadata& m) {
  out << "["
      << "time_index = " << m.time_index << ", "
      << "nr_timesteps = " << m.nr_timesteps << ", "
      << "channel_begin = " << m.channel_begin << ", "
      << "channel_end = " << m.channel_end << ", "
      << "baseline = " << m.baseline << ", "
      << "coordinate = " << m.coordinate << ", "
      << "nr_aterms = " << m.nr_aterms << "]";
  return out;
}

template <class T>
ostream& operator<<(ostream& out, UVW<T>& uvw) {
  out << "(" << uvw.u << "," << uvw.v << "," << uvw.w << ")";
  return out;
}

ostream& operator<<(ostream& os, const float2& x) {
  os << "(" << x.real << "," << x.imag << ")";
  return os;
}

ostream& operator<<(ostream& os, const double2& x) {
  os << "(" << x.real << "," << x.imag << ")";
  return os;
}

}  // namespace idg
