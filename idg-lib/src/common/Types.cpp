// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Types.h"

using std::ostream;

namespace idg {

ostream& operator<<(ostream& out, const Baseline& b) {
  out << "(" << b.station1 << "," << b.station2 << ")";
  return out;
}

ostream& operator<<(ostream& out, const Coordinate& c) {
  out << "(" << c.x << "," << c.y << ")";
  return out;
}

ostream& operator<<(ostream& out, const Metadata& m) {
  out << "["
      << "time_index = " << m.time_index << ", "
      << "nr_timesteps = " << m.nr_timesteps << ", "
      << "channel_begin = " << m.channel_begin << ", "
      << "channel_end = " << m.channel_end << ", "
      << "baseline = " << m.baseline << ", "
      << "coordinate = " << m.coordinate << ", "
      << "wtile_coordinate = " << m.wtile_coordinate << ", "
      << "wtile_index = " << m.wtile_index << ", "
      << "nr_aterms = " << m.nr_aterms << "]";
  return out;
}

template <class T>
ostream& operator<<(ostream& out, const UVW<T>& uvw) {
  out << "(" << uvw.u << "," << uvw.v << "," << uvw.w << ")";
  return out;
}

}  // namespace idg
