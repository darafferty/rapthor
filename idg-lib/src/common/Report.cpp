// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Report.h"

#include "idg-config.h"

using namespace std;

namespace idg {

#define FW1 12
#define FW2 8

void report(string name, double runtime) {
#if defined(PERFORMANCE_REPORT)
  clog << setw(FW1) << left << string(name) + ": " << setw(FW2) << right
       << scientific << setprecision(4) << runtime << " s" << endl;
#endif
}

void report(string name, double runtime, double joules, uint64_t flops,
            uint64_t bytes, bool ignore_short) {
#if defined(PERFORMANCE_REPORT)
  // Ignore very short measurements
  if (ignore_short && runtime < 1e-3) {
    return;
  }

  // Ignore unrealistic performance
  int gflops_bound = 1e5;  // 100 TFLOPS

  double watt = joules / runtime;
#pragma omp critical(clog)
  {
    clog << setw(FW1) << left << string(name) + ": " << setw(FW2) << right
         << scientific << setprecision(4) << runtime << " s";
    if (flops != 0) {
      clog << ", ";
      double gflops = (flops / runtime) * 1e-9;
      if (gflops < gflops_bound) {
        clog << setw(FW2) << right << fixed << setprecision(2) << gflops
             << " GFLOPS";
      }
    }
    if (bytes != 0) {
      clog << ", ";
      clog << setw(FW2) << right << fixed << setprecision(2)
           << bytes / runtime * 1e-9 << " GB/s";
    }
    if (watt > 1) {
      clog << ", ";
      clog << setw(FW2) << right << fixed << setprecision(2) << watt << " Watt";
    }
    if (flops != 0 && watt > 1) {
      clog << ", ";
      clog << setw(FW2) << right << fixed << setprecision(2)
           << (flops / runtime * 1e-9) / watt << " GFLOPS/W";
    }
    if (joules > 1) {
      clog << ", ";
      clog << setw(FW2) << right << fixed << setprecision(2) << joules
           << " Joules";
    }
  }
  clog << endl;
#endif
}

void report(string name, uint64_t flops, uint64_t bytes,
            powersensor::PowerSensor* powerSensor,
            powersensor::State startState, powersensor::State endState) {
  double seconds = powerSensor->seconds(startState, endState);
  double joules = powerSensor->Joules(startState, endState);
  bool ignore_short = false;
  report(name, seconds, flops, bytes, joules, ignore_short);
  return;
}

void report_visibilities(string name, double runtime,
                         uint64_t nr_visibilities) {
#if defined(PERFORMANCE_REPORT)
  clog << setw(FW1) << left << string(name) + ": " << fixed << setprecision(2)
       << 1e-6 * nr_visibilities / runtime << " Mvisibilities/s" << endl;
#endif
}

void Report::update(Report::State& reportState, powersensor::State& startState,
                    powersensor::State& endState) {
  reportState.current_seconds =
      powersensor::PowerSensor::seconds(startState, endState);
  reportState.current_joules =
      powersensor::PowerSensor::Joules(startState, endState);
  reportState.total_seconds += reportState.current_seconds;
  reportState.total_joules += reportState.current_joules;
}

void Report::update_devices(std::vector<powersensor::State> start,
                            std::vector<powersensor::State> end) {
  assert(start.size() == end.size());
  for (unsigned d = 0; d < start.size(); d++) {
    update_device(start[d], end[d], d);
  }
}

void Report::print_device(powersensor::State& startState,
                          powersensor::State& endState, int i) {
  std::stringstream name;
  name << prefix << auxiliary::name_device;
  if (i > 0) {
    name << i;
  }
  double seconds = powersensor::PowerSensor::seconds(startState, endState);
  double joules = powersensor::PowerSensor::Joules(startState, endState);
  report(name.str().c_str(), seconds, joules, 0, 0, false);
}

void Report::print_devices(std::vector<powersensor::State> start,
                           std::vector<powersensor::State> end) {
  assert(start.size() == end.size());
  for (unsigned i = 0; i < start.size(); i++) {
    print_device(start[i], end[i], i);
  }
}

}  // end namespace idg
