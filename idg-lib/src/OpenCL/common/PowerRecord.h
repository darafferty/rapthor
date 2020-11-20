// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdio>

#include <CL/cl.hpp>

#include "idg-common.h"

namespace idg {
namespace kernel {
namespace opencl {

class PowerRecord {
 public:
  PowerRecord();
  PowerRecord(powersensor::PowerSensor *sensor);

  void enqueue(cl::CommandQueue &queue);
  static void getPower(cl_event, cl_int, void *userData);
  powersensor::PowerSensor *sensor;
  powersensor::State state;
  cl::Event event;
};

}  // namespace opencl
}  // end namespace kernel
}  // end namespace idg
