// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <vector>

#include "omp.h"

#include "auxiliary.h"
#include "idg-config.h"

#if defined(HAVE_PMT)
#include <pmt.h>
#endif

#include "Pmt.h"

namespace idg::pmt {

static std::string name_likwid("likwid");
static std::string name_rapl("rapl");
static std::string name_nvml("nvml");
static std::string name_arduino("tty");

class PmtImpl : public Pmt {
 public:
#if defined(HAVE_PMT)
  explicit PmtImpl(const std::string& name, size_t device_number);
#else
  PmtImpl() = default;
#endif
  State Read() override;

 private:
#if defined(HAVE_PMT)
  std::unique_ptr<::pmt::PMT> power_meter_;
#endif
};

#if defined(HAVE_PMT)
PmtImpl::PmtImpl(const std::string& name, size_t device_number) {
  // Determine which environment variable to read
  std::string power_meter_env_str;
  if (name.compare(sensor_host) == 0) {
    power_meter_env_str = sensor_host;
  } else if (name.compare(sensor_device) == 0) {
    power_meter_env_str = sensor_device;
  } else {
    power_meter_env_str = sensor_default;
  }

  // Read environment variable
  char* power_meter_char = getenv(power_meter_env_str.c_str());
  if (!power_meter_char) {
    power_meter_char = getenv(sensor_default.c_str());
  }

  // Split environment variable value
  std::vector<std::string> power_meter_strings =
      idg::auxiliary::split_string(power_meter_char, ",");

  // Try to initialize the specified PMT
  if (device_number < power_meter_strings.size()) {
    const std::string& power_meter_string = power_meter_strings[device_number];
#if defined(HAVE_LIKWID)
    if (power_meter_string.compare(name_likwid) == 0) {
      power_meter_ = ::Likwid::create();
    }
#endif
    if (power_meter_string.compare(name_rapl) == 0) {
      power_meter_ = ::pmt::rapl::Rapl::create();
    }
#if defined(HAVE_NVML)
    if (power_meter_string.compare(name_nvml) == 0) {
      power_meter_ = ::pmt::nvml::NVML::create(device_number);
    }
#endif
#if defined(HAVE_ARDUINO)
    if (power_meter_string.find(name_arduino) != std::string::npos) {
      power_meter_ =
          ::pmt::arduino::Arduino::create(power_meter_string.c_str());
    }
#endif
  }
}
#endif  // HAVE_PMT

State PmtImpl::Read() {
#if defined(HAVE_PMT)
  ::pmt::State state = power_meter_->read();
  return State{state.timeAtRead, state.joulesAtRead};
#else
  return State{omp_get_wtime(), 0};
#endif
}

double Pmt::Seconds(const State& first_state, const State& second_state) {
  return second_state.time_at_read - first_state.time_at_read;
}

double Pmt::Joules(const State& first_state, const State& second_state) {
  return second_state.joules_at_read - first_state.joules_at_read;
}

double Pmt::Watts(const State& first_state, const State& second_state) {
  return Joules(first_state, second_state) / Seconds(first_state, second_state);
}

std::unique_ptr<Pmt> get_power_meter([[maybe_unused]] const std::string& name,
                                     [[maybe_unused]] size_t device_number) {
#if defined(HAVE_PMT)
  return std::make_unique<PmtImpl>(name, device_number);
#else
  return std::make_unique<PmtImpl>();
#endif
}
}  // end namespace idg::pmt
