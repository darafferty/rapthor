// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cudawrappers/cu.hpp>

#include "PowerRecord.h"

namespace idg {
namespace kernel {
namespace cuda {

PowerRecord::PowerRecord(cu::Event& event, pmt::Pmt& sensor)
    : sensor(sensor), event(event) {}

void getPower(CUstream, CUresult, void* userData) {
  PowerRecord* record = static_cast<PowerRecord*>(userData);
  record->state = record->sensor.Read();
}

void PowerRecord::enqueue(cu::Stream& stream) {
  stream.record(event);
  stream.addCallback(&getPower, this);
}

}  // end namespace cuda
}  // end namespace kernel
}  // end namespace idg
