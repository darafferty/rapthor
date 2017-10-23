namespace idg{

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
            int nr_channels;
            int subgrid_size;
            int grid_size;
        };

        public:
            Report(
                const int nr_channels,
                const int subgrid_size,
                const int grid_size)
            {
                parameters.nr_channels  = nr_channels;
                parameters.subgrid_size = subgrid_size;
                parameters.grid_size = grid_size;
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
                powersensor::State& endState)
            {
                 reportState.current_seconds = powersensor::DummyPowerSensor::seconds(startState, endState);
                 reportState.current_joules  = powersensor::DummyPowerSensor::Joules(startState, endState);
                 reportState.total_seconds  += reportState.current_seconds;
                 reportState.total_joules   += reportState.current_joules;
            }

            void update_host(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                host_enabled = true;
                update(state_host, startState, endState);
            }

            void update_gridder(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                gridder_enabled = true;
                update(state_gridder, startState, endState);
            }

            void update_degridder(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                degridder_enabled = true;
                update(state_degridder, startState, endState);
            }

            void update_adder(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                adder_enabled = true;
                update(state_adder, startState, endState);
            }

            void update_splitter(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                splitter_enabled = true;
                update(state_splitter, startState, endState);
            }

            void update_scaler(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                scaler_enabled = true;
                update(state_scaler, startState, endState);
            }

            void update_subgrid_fft(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                subgrid_fft_enabled = true;
                update(state_subgrid_fft, startState, endState);
            }

            void update_grid_fft(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                grid_fft_enabled = true;
                update(state_grid_fft, startState, endState);
            }

            void update_fft_shift(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                fft_shift_enabled = true;
                update(state_fft_shift, startState, endState);
            }

            void update_fft_shift(
                double runtime)
            {
                fft_shift_enabled = true;
                update(state_fft_shift, runtime);
            }

            void update_fft_scale(
                double runtime)
            {
                fft_scale_enabled = true;
                update(state_fft_scale, runtime);
            }

            void update_input(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                input_enabled = true;
                update(state_input, startState, endState);
            }

            void update_output(
                powersensor::State& startState,
                powersensor::State& endState)
            {
                output_enabled = true;
                update(state_output, startState, endState);
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

                if (gridder_enabled) {
                auxiliary::report(
                    prefix + auxiliary::name_gridder,
                    total ? state_gridder.total_seconds : state_gridder.current_seconds,
                    auxiliary::flops_gridder(nr_channels, nr_timesteps, nr_subgrids, subgrid_size),
                    auxiliary::bytes_gridder(nr_channels, nr_timesteps, nr_subgrids, subgrid_size),
                    total ? state_gridder.total_joules : state_gridder.current_joules);
                }
                if (degridder_enabled) {
                    auxiliary::report(
                        prefix + auxiliary::name_degridder,
                        total ? state_degridder.total_seconds : state_degridder.current_seconds,
                        auxiliary::flops_degridder(nr_channels, nr_timesteps, nr_subgrids, subgrid_size),
                        auxiliary::bytes_degridder(nr_channels, nr_timesteps, nr_subgrids, subgrid_size),
                        total ? state_degridder.total_joules : state_degridder.current_joules);
                }
                 if (subgrid_fft_enabled) {
                    auxiliary::report(
                        prefix + auxiliary::name_subgrid_fft,
                        total ? state_subgrid_fft.total_seconds : state_subgrid_fft.current_seconds,
                        auxiliary::flops_fft(subgrid_size, nr_subgrids),
                        auxiliary::bytes_fft(subgrid_size, nr_subgrids),
                        total ? state_subgrid_fft.total_joules : state_subgrid_fft.current_joules);
                }
                if (adder_enabled) {
                    auxiliary::report(
                        prefix + auxiliary::name_adder,
                        total ? state_adder.total_seconds : state_adder.current_seconds,
                        auxiliary::flops_adder(nr_subgrids, subgrid_size),
                        auxiliary::bytes_adder(nr_subgrids, subgrid_size),
                        total ? state_adder.total_joules : state_adder.current_joules);
                }
                if (splitter_enabled) {
                    auxiliary::report(
                        prefix + auxiliary::name_splitter,
                        total ? state_splitter.total_seconds : state_splitter.current_seconds,
                        auxiliary::flops_splitter(nr_subgrids, subgrid_size),
                        auxiliary::bytes_splitter(nr_subgrids, subgrid_size),
                        total ? state_splitter.total_joules : state_splitter.current_joules);
                }
                if (scaler_enabled) {
                    auxiliary::report(
                        prefix + auxiliary::name_scaler,
                        total ? state_scaler.total_seconds : state_scaler.current_seconds,
                        auxiliary::flops_scaler(nr_subgrids, subgrid_size),
                        auxiliary::bytes_scaler(nr_subgrids, subgrid_size),
                        total ? state_scaler.total_joules : state_scaler.current_joules);
                }
                if (grid_fft_enabled) {
                    auxiliary::report(
                        prefix + auxiliary::name_grid_fft,
                        total ? state_grid_fft.total_seconds : state_grid_fft.current_seconds,
                        auxiliary::flops_fft(grid_size, 1),
                        auxiliary::bytes_fft(grid_size, 1),
                        total ? state_grid_fft.total_joules : state_grid_fft.current_joules);
                }
                if (fft_shift_enabled) {
                    auxiliary::report(
                        prefix + auxiliary::name_fft_shift,
                        total ? state_fft_shift.total_seconds : state_fft_shift.current_seconds,
                        0, 0, 0);
                }
                if (fft_scale_enabled) {
                    auxiliary::report(
                        prefix + auxiliary::name_fft_scale,
                        total ? state_fft_scale.total_seconds : state_fft_scale.current_seconds,
                        0, 0, 0);
                }
                if (host_enabled) {
                    auxiliary::report(
                        prefix + auxiliary::name_host,
                        total ? state_host.total_seconds : state_host.current_seconds,
                        0, 0,
                        total ? state_host.total_joules : state_host.current_joules);
                    }
            }

            void print_total(
                int nr_timesteps = 0,
                int nr_subgrids = 0)
            {
                print(nr_timesteps, nr_subgrids, true, prefix);
            }

            void print_visibilities(
                const std::string name,
                int nr_visibilities)
            {
                auxiliary::report_visibilities(
                    prefix + name,
                    state_host.total_seconds,
                    nr_visibilities);
            }

            void print_device(
                powersensor::State& startState,
                powersensor::State& endState,
                int i = -1)
            {
                    std::stringstream name;
                    name << prefix << auxiliary::name_device;
                    if (i > 0) {
                       name <<  i;
                    }
                    double seconds = powersensor::DummyPowerSensor::seconds(startState, endState);
                    double joules  = powersensor::DummyPowerSensor::Joules(startState, endState);
                    auxiliary::report(name.str().c_str(), seconds, 0, 0, joules);

            }
            void print_devices(
                std::vector<powersensor::State> start,
                std::vector<powersensor::State> end)
            {
                assert(start.size() == end.size());
                for (int i = 0; i < start.size(); i++) {
                    print_device(start[i], end[i], i);
                }
            }

        private:
            const std::string prefix = "|";

            bool host_enabled        = false;
            bool gridder_enabled     = false;
            bool degridder_enabled   = false;
            bool adder_enabled       = false;
            bool splitter_enabled    = false;
            bool scaler_enabled      = false;
            bool subgrid_fft_enabled = false;
            bool grid_fft_enabled       = false;
            bool fft_shift_enabled = false;
            bool fft_scale_enabled = false;
            bool input_enabled  = false;
            bool output_enabled = false;

            Parameters parameters;

            State state_host;
            State state_gridder;
            State state_degridder;
            State state_adder;
            State state_splitter;
            State state_scaler;
            State state_subgrid_fft;
            State state_grid_fft;
            State state_fft_shift;
            State state_fft_scale;
            State state_input;
            State state_output;
    };

} // end namespace idg
