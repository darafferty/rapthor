// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <vector>
#include <iostream>

#include "omp.h"

#include "idg-config.h"

#include "auxiliary.h"

#if defined(HAVE_POWERSENSOR)
#include "powersensor/LikwidPowerSensor.h"
#include "powersensor/RaplPowerSensor.h"
#include "powersensor/NVMLPowerSensor.h"
#include "powersensor/ArduinoPowerSensor.h"
#include "powersensor/AMDGPUPowerSensor.h"
#define POWERSENSOR_DEFINED
#endif

#include "PowerSensor.h"

namespace powersensor {

static std::string name_likwid("likwid");
static std::string name_rapl("rapl");
static std::string name_nvml("nvml");
static std::string name_arduino("tty");
static std::string name_amdgpu("amdgpu");

class DummyPowerSensor : public PowerSensor {
 public:
  DummyPowerSensor();
  virtual State read();
};

PowerSensor *get_power_sensor(const std::string name, const unsigned i) {
  // Determine which environment variable to read
  std::string power_sensor_env_str;
  if (name.compare(sensor_host) == 0) {
    power_sensor_env_str = sensor_host;
  } else if (name.compare(sensor_device) == 0) {
    power_sensor_env_str = sensor_device;
  } else {
    power_sensor_env_str = sensor_default;
  }

  // Read environment variable
  char *power_sensor_char = getenv(power_sensor_env_str.c_str());
  if (!power_sensor_char) {
    power_sensor_char = getenv(sensor_default.c_str());
  }

  // Split environment variable value
  std::vector<std::string> power_sensors =
      idg::auxiliary::split_string(power_sensor_char, ",");

// Try to initialize the specified PowerSensor
#if defined(HAVE_POWERSENSOR)
  if (power_sensors.size() > 0 && i < power_sensors.size()) {
    std::string power_sensor_str = power_sensors[i];
    if (power_sensor_str.compare(name_likwid) == 0) {
      return likwid::LikwidPowerSensor::create();
    } else if (power_sensor_str.compare(name_rapl) == 0) {
      return rapl::RaplPowerSensor::create(NULL);
    } else if (power_sensor_str.compare(name_nvml) == 0) {
      return nvml::NVMLPowerSensor::create(i, NULL);
    } else if (power_sensor_str.find(name_arduino) != std::string::npos) {
      return arduino::ArduinoPowerSensor::create(power_sensor_str.c_str(),
                                                 NULL);
    } else if (power_sensor_str.compare(name_amdgpu) == 0) {
      return amdgpu::AMDGPUPowerSensor::create(i, NULL);
    }
  }
#endif

  // Use the DummyPowerSensor as backup
  return new DummyPowerSensor();
}

PowerSensor::~PowerSensor(){};

DummyPowerSensor::DummyPowerSensor() {}

State DummyPowerSensor::read() {
  State state;
  state.timeAtRead = omp_get_wtime();
  return state;
}

double PowerSensor::seconds(const State &firstState, const State &secondState) {
  return secondState.timeAtRead - firstState.timeAtRead;
}

double PowerSensor::Joules(const State &firstState, const State &secondState) {
  return secondState.joulesAtRead - firstState.joulesAtRead;
}

double PowerSensor::Watt(const State &firstState, const State &secondState) {
  return Joules(firstState, secondState) / seconds(firstState, secondState);
}
}  // end namespace powersensor
