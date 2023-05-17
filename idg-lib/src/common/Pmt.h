// Copyright (C) 2023 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_PMT_H_
#define IDG_PMT_H_

#include <memory>
#include <string>

namespace idg::pmt {

static std::string sensor_default("POWER_SENSOR");
static std::string sensor_host("HOST_SENSOR");
static std::string sensor_device("DEVICE_SENSOR");

struct State {
  double time_at_read = 0;
  double joules_at_read = 0;
};

class Pmt {
 public:
  virtual State Read() = 0;
  static double Seconds(const State& first_state, const State& second_state);
  static double Joules(const State& first_state, const State& second_state);
  static double Watts(const State& first_state, const State& second_state);
};

std::unique_ptr<Pmt> get_power_meter(const std::string& name,
                                     size_t device_number = 0);

}  // end namespace idg::pmt

#endif
