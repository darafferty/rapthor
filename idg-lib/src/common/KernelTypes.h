// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_KERNELTYPES_H_
#define IDG_KERNELTYPES_H_

typedef struct {
  int x, y, z;
} Coordinate;

typedef struct {
  unsigned int station1, station2;
} Baseline;

typedef struct {
  int time_index;
  int nr_timesteps;
  int channel_begin;
  int channel_end;
  Baseline baseline;
  Coordinate coordinate;
  Coordinate wtile_coordinate;
  int wtile_index;
  int nr_aterms;
} Metadata;

#ifndef __OPENCL_VERSION__
template <class T>
struct UVW {
  T u;
  T v;
  T w;
};
#else
typedef struct {
  float u;
  float v;
  float w;
} UVW;
#endif

#endif  // IDG_KERNELTYPES_H_