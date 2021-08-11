// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "InstanceCPU.h"

using namespace std;

namespace idg {
namespace kernel {
namespace cpu {

// Constructor
InstanceCPU::InstanceCPU() : KernelsInstance() {
#if defined(DEBUG)
  cout << __func__ << endl;
#endif

  m_powersensor.reset(powersensor::get_power_sensor(powersensor::sensor_host));
}

// Destructor
InstanceCPU::~InstanceCPU() {
#if defined(DEBUG)
  cout << __func__ << endl;
#endif
}

}  // namespace cpu
}  // namespace kernel
}  // namespace idg
