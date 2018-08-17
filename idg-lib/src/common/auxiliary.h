#ifndef IDG_AUX_H_
#define IDG_AUX_H_

#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "idg-config.h"


inline int min(int a, int b) {
    return a < b ? a : b;
}

inline int max(int a, int b) {
    return a > b ? a : b;
}

namespace powersensor {
    class PowerSensor;
    class State;
}

namespace idg {
    namespace auxiliary {
        #define NR_CORRELATIONS 4

        /*
            Operation and byte count
        */
        uint64_t flops_gridder(
            uint64_t nr_channels,
            uint64_t nr_timesteps,
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations = 4);

        uint64_t bytes_gridder(
            uint64_t nr_channels,
            uint64_t nr_timesteps,
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations = 4);

        uint64_t flops_degridder(
            uint64_t nr_channels,
            uint64_t nr_timesteps,
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations = 4);

        uint64_t bytes_degridder(
            uint64_t nr_channels,
            uint64_t nr_timesteps,
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations = 4);

        uint64_t flops_fft(
            uint64_t size,
            uint64_t batch,
            uint64_t nr_correlations = 4);

        uint64_t bytes_fft(
            uint64_t size,
            uint64_t batch,
            uint64_t nr_correlations = 4);

        uint64_t flops_adder(
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations = 4);

        uint64_t bytes_adder(
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations = 4);

        uint64_t flops_splitter(
            uint64_t nr_subgrids,
            uint64_t subgrid_size);

        uint64_t bytes_splitter(
            uint64_t nr_subgrids,
            uint64_t subgrid_size);

        uint64_t flops_scaler(
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations = 4);

        uint64_t bytes_scaler(
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations = 4);

        /*
            Sizeof routines
        */
        uint64_t sizeof_visibilities(
            unsigned int nr_baselines,
            unsigned int nr_timesteps,
            unsigned int nr_channels);

        uint64_t sizeof_uvw(
            unsigned int nr_baselines,
            unsigned int nr_timesteps);

        uint64_t sizeof_subgrids(
            unsigned int nr_subgrids,
            unsigned int subgrid_size,
            uint64_t nr_correlations = 4);

        uint64_t sizeof_metadata(
            unsigned int nr_subgrids);

        uint64_t sizeof_grid(
            unsigned int grid_size,
            uint64_t nr_correlations = 4);

        uint64_t sizeof_wavenumbers(
            unsigned int nr_channels);

        uint64_t sizeof_aterms(
            unsigned int nr_stations,
            unsigned int nr_timeslots,
            unsigned int subgrid_size,
            uint64_t nr_correlations = 4);

        uint64_t sizeof_spheroidal(
            unsigned int subgrid_size);

        uint64_t sizeof_avg_aterm_correction(
            unsigned int subgrid_size,
            uint64_t nr_correlations = 4);

        uint64_t sizeof_baselines(
            unsigned int nr_baselines);

        uint64_t sizeof_aterms_offsets(
            unsigned int nr_timeslots);

        /*
            Performance reporting
         */
        const std::string name_gridding("gridding");
        const std::string name_degridding("degridding");
        const std::string name_adding("|adding");
        const std::string name_splitting("|splitting");
        const std::string name_adder("adder");
        const std::string name_splitter("splitter");
        const std::string name_gridder("gridder");
        const std::string name_degridder("degridder");
        const std::string name_subgrid_fft("sub-fft");
        const std::string name_grid_fft("grid-fft");
        const std::string name_fft_shift("fft-shift");
        const std::string name_fft_scale("fft-scale");
        const std::string name_scaler("scaler");
        const std::string name_host("host");
        const std::string name_device("device");

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

        /*
            Misc
        */
        std::vector<int> split_int(const char *string, const char *delimiter);
        std::vector<std::string> split_string(char *string, const char *delimiter);

        std::string get_lib_dir();
    } // namespace auxiliary
} // namespace idg

#endif
