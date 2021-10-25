// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef IDG_REPORT_H_
#define IDG_REPORT_H_

#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cassert>

#include "auxiliary.h"

#include "PowerSensor.h"

namespace idg {

namespace auxiliary {
/*
    Strings
*/
const std::string name_gridding("gridding");
const std::string name_degridding("degridding");
const std::string name_adding("|adding");
const std::string name_splitting("|splitting");
const std::string name_adder("adder");
const std::string name_splitter("splitter");
const std::string name_gridder("gridder");
const std::string name_degridder("degridder");
const std::string name_calibrate("calibrate");
const std::string name_subgrid_fft("sub-fft");
const std::string name_grid_fft("grid-fft");
const std::string name_fft_shift("fft-shift");
const std::string name_fft_scale("fft-scale");
const std::string name_scaler("scaler");
const std::string name_average_beam("average-beam");
const std::string name_wtiling_forward("wtiling");
const std::string name_wtiling_backward("iwtiling");
const std::string name_host("host");
const std::string name_device("device");
}  // namespace auxiliary

/*
    Performance reporting
 */
void report(const std::string name, double runtime);

void report(const std::string name, double runtime, double joules,
            uint64_t flops, uint64_t bytes, bool ignore_short = false);

void report(const std::string name, uint64_t flops, uint64_t bytes,
            powersensor::PowerSensor* powerSensor,
            powersensor::State startState, powersensor::State endState);

void report_visibilities(const std::string name, double runtime,
                         uint64_t nr_visibilities);

class Report {
  struct State {
    double current_seconds = 0;
    double current_joules = 0;
    double total_seconds = 0;
    double total_joules = 0;
  };

  struct Parameters {
    int nr_channels = 0;
    int subgrid_size = 0;
    int grid_size = 0;
    int nr_terms = 0;
    int nr_timesteps = 0;
    int nr_subgrids = 0;
    int nr_correlations = 0;
  };

  struct Counters {
    int total_nr_subgrids = 0;
    int total_nr_timesteps = 0;
    int total_nr_visibilities = 0;
  };

 public:
  enum ID {
    gridding,
    degridding,
    adding,
    splitting,
    adder,
    splitter,
    gridder,
    degridder,
    calibrate,
    subgrid_fft,
    grid_fft,
    fft_shift,
    fft_scale,
    average_beam,
    wtiling_forward,
    wtiling_backward,
    host,
    device,
    sentinel
  };

  inline std::string get_name(ID id) {
    switch (id) {
      case gridding:
        return auxiliary::name_gridding;
      case degridding:
        return auxiliary::name_degridder;
      case adding:
        return auxiliary::name_adding;
      case splitting:
        return auxiliary::name_splitting;
      case adder:
        return auxiliary::name_adder;
      case splitter:
        return auxiliary::name_splitter;
      case gridder:
        return auxiliary::name_gridder;
      case degridder:
        return auxiliary::name_degridder;
      case calibrate:
        return auxiliary::name_calibrate;
      case subgrid_fft:
        return auxiliary::name_subgrid_fft;
      case grid_fft:
        return auxiliary::name_grid_fft;
      case fft_shift:
        return auxiliary::name_fft_shift;
      case fft_scale:
        return auxiliary::name_fft_scale;
      case average_beam:
        return auxiliary::name_average_beam;
      case wtiling_forward:
        return auxiliary::name_wtiling_forward;
      case wtiling_backward:
        return auxiliary::name_wtiling_backward;
      case host:
        return auxiliary::name_host;
      case device:
        return auxiliary::name_device;
      default:
        return std::string("unknown");
    }
  }

  inline uint64_t get_flops(ID id, Parameters parameters) {
    int nr_channels = parameters.nr_channels;
    int subgrid_size = parameters.subgrid_size;
    int grid_size = parameters.grid_size;
    int nr_terms = parameters.nr_terms;
    int nr_timesteps = parameters.nr_timesteps;
    int nr_subgrids = parameters.nr_subgrids;
    int nr_correlations = parameters.nr_correlations;

    switch (id) {
      case adder:
        return auxiliary::flops_adder(nr_subgrids, subgrid_size,
                                      nr_correlations);
      case splitter:
        return auxiliary::flops_splitter(nr_subgrids, subgrid_size,
                                         nr_correlations);
      case gridder:
        return auxiliary::flops_gridder(nr_channels, nr_timesteps, nr_subgrids,
                                        subgrid_size, nr_correlations);
      case degridder:
        return auxiliary::flops_degridder(nr_channels, nr_timesteps,
                                          nr_subgrids, subgrid_size,
                                          nr_correlations);
      case calibrate:
        return auxiliary::flops_calibrate(nr_terms, nr_channels, nr_timesteps,
                                          nr_subgrids, subgrid_size);
      case subgrid_fft:
        return auxiliary::flops_fft(subgrid_size, nr_subgrids);
      case grid_fft:
        return auxiliary::flops_fft(grid_size, 1);
      default:
        return 0;
    }
  }

  inline uint64_t get_bytes(ID id, Parameters parameters) {
    int nr_channels = parameters.nr_channels;
    int subgrid_size = parameters.subgrid_size;
    int grid_size = parameters.grid_size;
    int nr_timesteps = parameters.nr_timesteps;
    int nr_subgrids = parameters.nr_subgrids;
    int nr_correlations = parameters.nr_correlations;

    switch (id) {
      case adder:
        return auxiliary::bytes_adder(nr_subgrids, subgrid_size,
                                      nr_correlations);
      case splitter:
        return auxiliary::bytes_splitter(nr_subgrids, subgrid_size,
                                         nr_correlations);
      case gridder:
        return auxiliary::bytes_gridder(nr_channels, nr_timesteps, nr_subgrids,
                                        subgrid_size, nr_correlations);
      case degridder:
        return auxiliary::bytes_degridder(nr_channels, nr_timesteps,
                                          nr_subgrids, subgrid_size,
                                          nr_correlations);
      case subgrid_fft:
        return auxiliary::bytes_fft(subgrid_size, nr_subgrids);
      case grid_fft:
        return auxiliary::bytes_fft(grid_size, 1);
      default:
        return 0;
    }
  }

  class ItemState {
   public:
    ItemState() { reset(); }

    void reset() {
      enabled = false;
      updated = false;
      runtime_current = 0;
      energy_current = 0;
      runtime_total = 0;
      energy_total = 0;
    }

    ID id;
    bool enabled = false;
    bool updated = false;
    double runtime_current = 0;
    double energy_current = 0;
    double runtime_total = 0;
    double energy_total = 0;
  };

 public:
  Report(const int nr_channels = 0, const int subgrid_size = 0,
         const int grid_size = 0, const int nr_terms = 0) {
    parameters.nr_channels = nr_channels;
    parameters.subgrid_size = subgrid_size;
    parameters.grid_size = grid_size;
    parameters.nr_terms = nr_terms;
    reset();
    int nr_items = ID::sentinel;
    items.resize(nr_items);
    for (int id = 0; id < nr_items; id++) {
      items[id].id = ID(id);
    }
  }

  void initialize(const int nr_channels = 0, const int subgrid_size = 0,
                  const int grid_size = 0, const int nr_terms = 0) {
    parameters.nr_channels = nr_channels;
    parameters.subgrid_size = subgrid_size;
    parameters.grid_size = grid_size;
    parameters.nr_terms = nr_terms;
    reset();
  }

  void update(ID id, double runtime) {
    auto& item = items[id];
    item.enabled = true;
    item.updated = true;
    item.runtime_current = runtime;
    item.runtime_total += runtime;
  }

  template <ID id>
  void update(double runtime) {
    update(id, runtime);
  }

  void update(ID id, powersensor::State& start, powersensor::State& end) {
    double runtime = powersensor::PowerSensor::seconds(start, end);
    double energy = powersensor::PowerSensor::Joules(start, end);
    update(id, runtime);
    auto& item = items[id];
    item.energy_current = energy;
    item.energy_total = energy;
  }

  template <ID id>
  void update(powersensor::State& start, powersensor::State& end) {
    update(id, start, end);
  }

  void update_total(int nr_subgrids, int nr_timesteps, int nr_visibilities) {
    counters.total_nr_subgrids += nr_subgrids;
    counters.total_nr_timesteps += nr_timesteps;
    counters.total_nr_visibilities += nr_visibilities;
  }

  void print(int nr_correlations, int nr_timesteps, int nr_subgrids,
             bool total = false, std::string prefix = "") {
    // Add additional parameters needed to compute flops and bytes
    parameters.nr_timesteps = nr_timesteps;
    parameters.nr_subgrids = nr_subgrids;
    parameters.nr_correlations = nr_correlations;

    // Do not report short measurements, unless reporting total runtime
    bool ignore_short = !total;

    for (auto& item : items) {
      if ((total && item.enabled) || item.updated) {
        auto seconds = total ? item.runtime_total : item.runtime_current;
        auto joules = total ? item.energy_total : item.energy_current;
        auto flops = get_flops(item.id, parameters);
        auto bytes = get_bytes(item.id, parameters);
        report(prefix + get_name(item.id), seconds, joules, flops, bytes,
               ignore_short);
        item.updated = false;
      }
    }
  }

  void print_total(int nr_correlations, int nr_timesteps = 0,
                   int nr_subgrids = 0) {
    parameters.nr_correlations = nr_correlations;
    if (nr_timesteps == 0) {
      nr_timesteps = counters.total_nr_timesteps;
    }
    if (nr_subgrids == 0) {
      nr_subgrids = counters.total_nr_subgrids;
    }
    print(nr_correlations, nr_timesteps, nr_subgrids, true, prefix);
  }

  void print_visibilities(const std::string name, int nr_visibilities = 0) {
    if (nr_visibilities == 0) {
      nr_visibilities = counters.total_nr_visibilities;
    }
    double runtime = items[host].runtime_total;
    report_visibilities(prefix + name, runtime, nr_visibilities);
  }

  void reset() {
    for (auto& item : items) {
      item.reset();
    }

    counters.total_nr_subgrids = 0;
    counters.total_nr_timesteps = 0;
    counters.total_nr_visibilities = 0;
  }

 private:
  const std::string prefix = "|";

  Parameters parameters;

  Counters counters;

  std::vector<ItemState> items;
};

}  // end namespace idg

#endif
