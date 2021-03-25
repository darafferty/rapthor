// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "PowerRecord.h"

namespace idg {
namespace kernel {
namespace cuda {

PowerRecord::PowerRecord(cu::Event &event, powersensor::PowerSensor &sensor)
    : sensor(sensor), event(event) {}

void PowerRecord::enqueue(cu::Stream &stream) {
  stream.record(event);
  stream.addCallback(&PowerRecord::getPower, this);
}

void PowerRecord::getPower(CUstream, CUresult, void *userData) {
  PowerRecord *record = static_cast<PowerRecord *>(userData);
  record->state = record->sensor.read();
}

}  // end namespace cuda
}  // end namespace kernel
}  // end namespace idg
