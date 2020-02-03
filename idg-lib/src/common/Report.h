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
        const std::string name_host("host");
        const std::string name_device("device");
    }

    /*
        Performance reporting
     */
    void report(
        const std::string name,
        double runtime);

    void report(
        const std::string name,
        double runtime,
        double joules,
        uint64_t flops,
        uint64_t bytes,
        bool ignore_short = false);

    void report(
        const std::string name,
        uint64_t flops,
        uint64_t bytes,
        powersensor::PowerSensor *powerSensor,
        powersensor::State startState,
        powersensor::State endState);

    void report_visibilities(
        const std::string name,
        double runtime,
        uint64_t nr_visibilities);

    class Report
    {
        struct State
        {
            double current_seconds = 0;
            double current_joules  = 0;
            double total_seconds   = 0;
            double total_joules    = 0;
        };

        struct Parameters
        {
            int nr_channels  = 0;
            int subgrid_size = 0;
            int grid_size    = 0;
            int nr_terms     = 0;
        };

        struct Counters
        {
            int total_nr_subgrids     = 0;
            int total_nr_timesteps    = 0;
            int total_nr_visibilities = 0;
        };

        public:
            Report(
                const int nr_channels  = 0,
                const int subgrid_size = 0,
                const int grid_size    = 0,
                const int nr_terms     = 0)
            {
                parameters.nr_channels  = nr_channels;
                parameters.subgrid_size = subgrid_size;
                parameters.grid_size    = grid_size;
                parameters.nr_terms     = nr_terms;
                reset();
            }

            void initialize(
                const int nr_channels  = 0,
                const int subgrid_size = 0,
                const int grid_size    = 0,
                const int nr_terms     = 0)
            {
                parameters.nr_channels  = nr_channels;
                parameters.subgrid_size = subgrid_size;
                parameters.grid_size    = grid_size;
                parameters.nr_terms     = nr_terms;
                reset();
            }

            void update(
                Report::State& reportState,
                double runtime)
            {
                reportState.current_seconds = runtime;
                reportState.total_seconds  += reportState.current_seconds;
            }

            void update(
                Report::State& reportState,
                powersensor::State& startState,
                powersensor::State& endState);

            void update_host(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                host_enabled = true;
                host_updated = true;
                update(state_host, startState, endState);
            }

            void update_gridder(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                gridder_enabled = true;
                gridder_updated = true;
                update(state_gridder, startState, endState);
            }

            void update_gridder_post(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                gridder_post_enabled = true;
                gridder_post_updated = true;
                update(state_gridder_post, startState, endState);
            }

            void update_degridder(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                degridder_enabled = true;
                degridder_updated = true;
                update(state_degridder, startState, endState);
            }

            void update_degridder_pre(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                degridder_pre_enabled = true;
                degridder_pre_updated = true;
                update(state_degridder_pre, startState, endState);
            }

            void update_calibrate(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                calibrate_enabled = true;
                calibrate_updated = true;
                update(state_calibrate, startState, endState);
            }

            void update_adder(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                adder_enabled = true;
                adder_updated = true;
                update(state_adder, startState, endState);
            }

            void update_splitter(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                splitter_enabled = true;
                splitter_updated = true;
                update(state_splitter, startState, endState);
            }

            void update_scaler(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                scaler_enabled = true;
                scaler_updated = true;
                update(state_scaler, startState, endState);
            }

            void update_subgrid_fft(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                subgrid_fft_enabled = true;
                subgrid_fft_updated = true;
                update(state_subgrid_fft, startState, endState);
            }

            void update_grid_fft(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                grid_fft_enabled = true;
                grid_fft_updated = true;
                update(state_grid_fft, startState, endState);
            }

            void update_fft_shift(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                fft_shift_enabled = true;
                fft_shift_updated = true;
                update(state_fft_shift, startState, endState);
            }

            void update_fft_shift(
                double runtime)
            {
                fft_shift_enabled = true;
                fft_shift_updated = true;
                update(state_fft_shift, runtime);
            }

            void update_fft_scale(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                fft_scale_enabled = true;
                fft_scale_updated = true;
                update(state_fft_scale, startState, endState);
            }

            void update_fft_scale(
                double runtime)
            {
                fft_scale_enabled = true;
                fft_scale_updated = true;
                update(state_fft_scale, runtime);
            }

            void update_input(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                input_enabled = true;
                input_updated = true;
                update(state_input, startState, endState);
            }

            void update_output(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                output_enabled = true;
                output_updated = true;
                update(state_output, startState, endState);
            }

            void update_device(
                powersensor::State& startState,
                powersensor::State& endState,
                unsigned id = 0)
            {
                if (states_device.size() <= id) {
                    State state;
                    states_device.push_back(state);
                }
                update(states_device[id], startState, endState);
            }

            void update_devices(
                std::vector<powersensor::State> start,
                std::vector<powersensor::State> end);

            void update_total(
                int nr_subgrids,
                int nr_timesteps,
                int nr_visibilities)
            {
                counters.total_nr_subgrids     += nr_subgrids;
                counters.total_nr_timesteps    += nr_timesteps;
                counters.total_nr_visibilities += nr_visibilities;
            }

            void print(
                int nr_timesteps,
                int nr_subgrids,
                bool total = false,
                std::string prefix = "")
            {
                auto nr_channels  = parameters.nr_channels;
                auto subgrid_size = parameters.subgrid_size;
                auto grid_size    = parameters.grid_size;
                auto nr_terms     = parameters.nr_terms;

                // Do not report short measurements, unless reporting total runtime
                bool ignore_short = !total;

                if ((total && gridder_enabled) || gridder_updated) {
                    auto seconds = total ? state_gridder.total_seconds : state_gridder.current_seconds;
                    auto joules  = total ? state_gridder.total_joules : state_gridder.current_joules;
                    if (gridder_post_updated) {
                        seconds += total ? state_gridder_post.total_seconds : state_gridder_post.current_seconds;
                        joules  += total ? state_gridder_post.total_joules : state_gridder_post.current_joules;
                    }
                    report(
                        prefix + auxiliary::name_gridder,
                        seconds,
                        joules,
                        auxiliary::flops_gridder(nr_channels, nr_timesteps, nr_subgrids, subgrid_size),
                        auxiliary::bytes_gridder(nr_channels, nr_timesteps, nr_subgrids, subgrid_size),
                        ignore_short);
                    gridder_updated = false;
                }
                if ((total && degridder_enabled) || degridder_updated) {
                    auto seconds = total ? state_degridder.total_seconds : state_degridder.current_seconds;
                    auto joules  = total ? state_degridder.total_joules : state_degridder.current_joules;
                    if (degridder_pre_updated) {
                        seconds += total ? state_degridder_pre.total_seconds : state_degridder_pre.current_seconds;
                        joules  += total ? state_degridder_pre.total_joules : state_degridder_pre.current_joules;
                    }
                    report(
                        prefix + auxiliary::name_degridder,
                        seconds,
                        joules,
                        auxiliary::flops_degridder(nr_channels, nr_timesteps, nr_subgrids, subgrid_size),
                        auxiliary::bytes_degridder(nr_channels, nr_timesteps, nr_subgrids, subgrid_size),
                        ignore_short);
                    degridder_updated = false;
                }
                if ((total && subgrid_fft_enabled) || subgrid_fft_updated) {
                    report(
                        prefix + auxiliary::name_subgrid_fft,
                        total ? state_subgrid_fft.total_seconds : state_subgrid_fft.current_seconds,
                        total ? state_subgrid_fft.total_joules : state_subgrid_fft.current_joules,
                        auxiliary::flops_fft(subgrid_size, nr_subgrids),
                        auxiliary::bytes_fft(subgrid_size, nr_subgrids),
                        ignore_short);
                    subgrid_fft_updated = false;
                }
                if ((total && calibrate_enabled) || calibrate_updated) {
                    report(
                        prefix + auxiliary::name_calibrate,
                        total ? state_calibrate.total_seconds : state_calibrate.current_seconds,
                        total ? state_calibrate.total_joules : state_calibrate.current_joules,
                        auxiliary::flops_calibrate(nr_terms, nr_channels, nr_timesteps, nr_subgrids, subgrid_size),
                        auxiliary::bytes_calibrate(),
                        ignore_short);
                    calibrate_updated = false;
                }
                if ((total && adder_enabled) || adder_updated) {
                    report(
                        prefix + auxiliary::name_adder,
                        total ? state_adder.total_seconds : state_adder.current_seconds,
                        total ? state_adder.total_joules : state_adder.current_joules,
                        auxiliary::flops_adder(nr_subgrids, subgrid_size),
                        auxiliary::bytes_adder(nr_subgrids, subgrid_size),
                        ignore_short);
                    adder_updated = false;
                }
                if ((total && splitter_enabled) || splitter_updated) {
                    report(
                        prefix + auxiliary::name_splitter,
                        total ? state_splitter.total_seconds : state_splitter.current_seconds,
                        total ? state_splitter.total_joules : state_splitter.current_joules,
                        auxiliary::flops_splitter(nr_subgrids, subgrid_size),
                        auxiliary::bytes_splitter(nr_subgrids, subgrid_size),
                        ignore_short);
                    splitter_updated = false;
                }
                if ((total && scaler_enabled) || scaler_updated) {
                    report(
                        prefix + auxiliary::name_scaler,
                        total ? state_scaler.total_seconds : state_scaler.current_seconds,
                        total ? state_scaler.total_joules : state_scaler.current_joules,
                        auxiliary::flops_scaler(nr_subgrids, subgrid_size),
                        auxiliary::bytes_scaler(nr_subgrids, subgrid_size),
                        ignore_short);
                    scaler_updated = false;
                }
                if ((total && grid_fft_updated) || grid_fft_updated) {
                    report(
                        prefix + auxiliary::name_grid_fft,
                        total ? state_grid_fft.total_seconds : state_grid_fft.current_seconds,
                        total ? state_grid_fft.total_joules : state_grid_fft.current_joules,
                        auxiliary::flops_fft(grid_size, 1),
                        auxiliary::bytes_fft(grid_size, 1),
                        ignore_short);
                    grid_fft_updated = false;
                }
                if ((total && fft_shift_enabled) || fft_shift_updated) {
                    report(
                        prefix + auxiliary::name_fft_shift,
                        total ? state_fft_shift.total_seconds : state_fft_shift.current_seconds,
                        0, 0, 0, ignore_short);
                    fft_shift_updated = false;
                }
                if ((total && fft_scale_enabled) || fft_scale_updated) {
                    report(
                        prefix + auxiliary::name_fft_scale,
                        total ? state_fft_scale.total_seconds : state_fft_scale.current_seconds,
                        0, 0, 0, ignore_short);
                    fft_scale_updated = false;
                }
                if ((total && host_enabled) || host_updated) {
                    report(
                        prefix + auxiliary::name_host,
                        total ? state_host.total_seconds : state_host.current_seconds,
                        total ? state_host.total_joules : state_host.current_joules,
                        0, 0, ignore_short);
                    host_updated = false;
                }
            }

            void print_total(
                int nr_timesteps = 0,
                int nr_subgrids = 0)
            {
                if (nr_timesteps == 0) {
                    nr_timesteps = counters.total_nr_timesteps;
                }
                if (nr_subgrids == 0) {
                    nr_subgrids = counters.total_nr_subgrids;
                }
                print(nr_timesteps, nr_subgrids, true, prefix);
            }

            void print_visibilities(
                const std::string name,
                int nr_visibilities = 0)
            {
                if (nr_visibilities == 0) {
                    nr_visibilities = counters.total_nr_visibilities;
                }
                report_visibilities(
                    prefix + name,
                    state_host.total_seconds,
                    nr_visibilities);
            }

            void print_device(
                powersensor::State& startState,
                powersensor::State& endState,
                int i = -1);

            void print_devices(
                std::vector<powersensor::State> start,
                std::vector<powersensor::State> end);

            void print_devices()
            {
                for (unsigned i = 0; i < states_device.size(); i++) {
                    State state = states_device[i];
                    std::stringstream name;
                    name << prefix << auxiliary::name_device;
                    if (states_device.size() > 1) {
                       name <<  i;
                    }
                    report(name.str().c_str(), state.total_seconds, state.total_joules, 0, 0);
                }
            }

            void reset() {
                host_enabled        = false;
                gridder_enabled     = false;
                gridder_post_enabled  = false;
                degridder_pre_enabled = false;
                degridder_enabled   = false;
                adder_enabled       = false;
                calibrate_enabled   = false;
                splitter_enabled    = false;
                scaler_enabled      = false;
                subgrid_fft_enabled = false;
                grid_fft_enabled    = false;
                fft_shift_enabled   = false;
                fft_scale_enabled   = false;
                input_enabled       = false;
                output_enabled      = false;

                host_updated        = false;
                gridder_updated     = false;
                gridder_post_updated  = false;
                degridder_pre_updated = false;
                degridder_updated   = false;
                adder_updated       = false;
                calibrate_updated   = false;
                splitter_updated    = false;
                scaler_updated      = false;
                subgrid_fft_updated = false;
                grid_fft_updated    = false;
                fft_shift_updated   = false;
                fft_scale_updated   = false;
                input_updated       = false;
                output_updated      = false;

                state_host        = state_zero;
                state_gridder     = state_zero;
                state_degridder   = state_zero;
                state_calibrate   = state_zero;
                state_adder       = state_zero;
                state_splitter    = state_zero;
                state_scaler      = state_zero;
                state_subgrid_fft = state_zero;
                state_grid_fft    = state_zero;
                state_fft_shift   = state_zero;
                state_fft_scale   = state_zero;
                state_input       = state_zero;
                state_output      = state_zero;
                states_device.clear();

                counters.total_nr_subgrids     = 0;
                counters.total_nr_timesteps    = 0;
                counters.total_nr_visibilities = 0;
            }

        private:
            const std::string prefix = "|";

            bool host_enabled;
            bool gridder_enabled;
            bool gridder_post_enabled;
            bool degridder_enabled;
            bool degridder_pre_enabled;
            bool calibrate_enabled;
            bool adder_enabled;
            bool splitter_enabled;
            bool scaler_enabled;
            bool subgrid_fft_enabled;
            bool grid_fft_enabled;
            bool fft_shift_enabled;
            bool fft_scale_enabled;
            bool input_enabled;
            bool output_enabled;

            bool host_updated;
            bool gridder_updated;
            bool gridder_post_updated;
            bool degridder_pre_updated;
            bool degridder_updated;
            bool calibrate_updated;
            bool adder_updated;
            bool splitter_updated;
            bool scaler_updated;
            bool subgrid_fft_updated;
            bool grid_fft_updated;
            bool fft_shift_updated;
            bool fft_scale_updated;
            bool input_updated;
            bool output_updated;

            Parameters parameters;

            const State state_zero;
            State state_host;
            State state_gridder;
            State state_gridder_post;
            State state_degridder;
            State state_degridder_pre;
            State state_calibrate;
            State state_adder;
            State state_splitter;
            State state_scaler;
            State state_subgrid_fft;
            State state_grid_fft;
            State state_fft_shift;
            State state_fft_scale;
            State state_input;
            State state_output;
            std::vector<State> states_device;

            Counters counters;
    };

} // end namespace idg

#endif
