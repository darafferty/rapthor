#ifndef IDG_AUX_H_
#define IDG_AUX_H_

#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

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

        uint64_t flops_calibrate(
            uint64_t nr_terms,
            uint64_t nr_channels,
            uint64_t nr_timesteps,
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations = 4);

        uint64_t bytes_calibrate();

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

        uint64_t sizeof_aterms_indices(
            unsigned int nr_baselines,
            unsigned int nr_timesteps);

        uint64_t sizeof_spheroidal(
            unsigned int subgrid_size);

        uint64_t sizeof_avg_aterm_correction(
            unsigned int subgrid_size,
            uint64_t nr_correlations = 4);

        uint64_t sizeof_baselines(
            unsigned int nr_baselines);

        uint64_t sizeof_aterms_offsets(
            unsigned int nr_timeslots);

        uint64_t sizeof_weights(
            unsigned int nr_baselines,
            unsigned int nr_timesteps,
            unsigned int nr_channels,
            unsigned int nr_correlations = 4);

        /*
            Misc
        */
        std::vector<int> split_int(const char *string, const char *delimiter);
        std::vector<std::string> split_string(char *string, const char *delimiter);

        std::string get_lib_dir();

        size_t get_total_memory();
        size_t get_used_memory();

    } // namespace auxiliary
} // namespace idg

#endif
