#include <iostream>
#include <iomanip>
#include <cstdint>
#include <omp.h>

#include "idg-config.h"
#include "idg-common.h"
#include "auxiliary.h"
#include "PowerSensor.h"

using namespace std;

namespace idg {
    namespace auxiliary {

        /*
            Operation and byte count
        */
        uint64_t flops_gridder(
            uint64_t nr_channels,
            uint64_t nr_timesteps,
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations)
        {
            // Number of flops per visibility
            uint64_t flops_per_visibility = 0;
            flops_per_visibility += 5; // phase index
            flops_per_visibility += 5; // phase offset
            flops_per_visibility += nr_channels * 2; // phase
            #if defined(REPORT_OPS)
            flops_per_visibility += nr_channels * 2; // phasor
            #endif
            flops_per_visibility += nr_channels * nr_correlations * 8; // update

            // Number of flops per subgrid
            uint64_t flops_per_subgrid = 0;
            flops_per_subgrid += nr_correlations * 30; // aterm
            flops_per_subgrid += nr_correlations * 2; // spheroidal
            flops_per_subgrid += 6; // shift

            // Total number of flops
            uint64_t flops_total = 0;
            flops_total += nr_timesteps * subgrid_size * subgrid_size * flops_per_visibility;
            flops_total += nr_subgrids  * subgrid_size * subgrid_size * flops_per_subgrid;
            return flops_total;
        }

        uint64_t bytes_gridder(
            uint64_t nr_channels,
            uint64_t nr_timesteps,
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations)
        {
            // Number of bytes per uvw coordinate
            uint64_t bytes_per_uvw = 0;
            bytes_per_uvw += 1ULL * 3 * sizeof(float); // read uvw

            // Number of bytes per visibility
            uint64_t bytes_per_vis = 0;
            bytes_per_vis += 1ULL * nr_channels * nr_correlations * 2 * sizeof(float); // read visibilities

            // Number of bytes per pixel
            uint64_t bytes_per_pix = 0;
            bytes_per_pix += 1ULL * nr_correlations * 2 * sizeof(float); // read pixel
            bytes_per_pix += 1ULL * nr_correlations * 2 * sizeof(float); // write pixel

            // Number of bytes per aterm
            uint64_t bytes_per_aterm = 0;
            bytes_per_aterm += 1ULL * 2 * nr_correlations * 2 * sizeof(float); // read aterm

            // Number of bytes per spheroidal
            uint64_t bytes_per_spheroidal = 0;
            bytes_per_spheroidal += 1ULL * sizeof(float); // read spheroidal

            // Total number of bytes
            uint64_t bytes_total = 0;
            bytes_total += 1ULL * nr_timesteps * bytes_per_uvw;
            bytes_total += 1ULL * nr_timesteps * bytes_per_vis;
            bytes_total += 1ULL * nr_subgrids * subgrid_size * subgrid_size * bytes_per_pix;
            bytes_total += 1ULL * nr_subgrids * subgrid_size * subgrid_size * bytes_per_aterm;
            bytes_total += 1ULL * nr_subgrids * subgrid_size * subgrid_size * bytes_per_spheroidal;
            return bytes_total;
        }

        uint64_t flops_degridder(
            uint64_t nr_channels,
            uint64_t nr_timesteps,
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations)
        {
            return flops_gridder(nr_channels, nr_timesteps, nr_subgrids, subgrid_size, nr_correlations);
        }

        uint64_t bytes_degridder(
            uint64_t nr_channels,
            uint64_t nr_timesteps,
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations)
        {
            return bytes_gridder(nr_channels, nr_timesteps, nr_subgrids, subgrid_size, nr_correlations);
        }

        uint64_t flops_fft(
            uint64_t size,
            uint64_t batch,
            uint64_t nr_correlations)
        {
            // Pseudo number of flops:
            // return 1ULL * 5 * batch * nr_correlations * size * size * log2(size * size);
            // Estimated number of flops based on fftwf_flops, which seems to
            // return the number of simd instructions, not scalar flops.
            return 1ULL * 4 * batch * nr_correlations * size * size * log2(size * size);
        }

        uint64_t bytes_fft(
            uint64_t size,
            uint64_t batch,
            uint64_t nr_correlations)
        {
            return 1ULL * 2 * batch * nr_correlations * size * size * 2 * sizeof(float);
        }

        uint64_t flops_adder(
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations)
        {
            uint64_t flops = 0;
            flops += 1ULL * nr_subgrids * subgrid_size * subgrid_size * 8; // shift
            flops += 1ULL * nr_subgrids * subgrid_size * subgrid_size * nr_correlations * 2; // add
            return flops;
        }

        uint64_t bytes_adder(
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations)
        {
            uint64_t bytes = 0;
            bytes += 1ULL * nr_subgrids * 2 * sizeof(int); // coordinate
            bytes += 1ULL * nr_subgrids * subgrid_size * subgrid_size * 2 * sizeof(float); // grid in
            bytes += 1ULL * nr_subgrids * subgrid_size * subgrid_size * 2 * sizeof(float); // subgrid in
            bytes += 1ULL * nr_subgrids * subgrid_size * subgrid_size * 2 * sizeof(float); // subgrid out
            return bytes;
        }

        uint64_t flops_splitter(
            uint64_t nr_subgrids,
            uint64_t subgrid_size)
        {
            uint64_t flops = 0;
            flops += 1ULL * nr_subgrids * subgrid_size * subgrid_size * 8; // shift
            return flops;
        }

        uint64_t bytes_splitter(
            uint64_t nr_subgrids,
            uint64_t subgrid_size)
        {
            uint64_t bytes = 0;
            bytes += 1ULL * nr_subgrids * 2 * sizeof(int); // coordinate
            bytes += 1ULL * nr_subgrids * subgrid_size * subgrid_size * 2 * sizeof(float); // grid in
            bytes += 1ULL * nr_subgrids * subgrid_size * subgrid_size * 2 * sizeof(float); // subgrid out
            return bytes;
        }

        uint64_t flops_scaler(
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations)
        {
            uint64_t flops = 0;
            flops += 1ULL * nr_subgrids * subgrid_size * subgrid_size * nr_correlations * 2; // scale
            return flops;
        }

        uint64_t bytes_scaler(
            uint64_t nr_subgrids,
            uint64_t subgrid_size,
            uint64_t nr_correlations)
        {
            uint64_t bytes = 0;
            bytes += 1ULL * nr_subgrids * subgrid_size * subgrid_size * nr_correlations * 2 * sizeof(float); // scale
            return bytes;
        }

        /*
            Sizeof routines
        */
        uint64_t sizeof_visibilities(
            unsigned int nr_baselines,
            unsigned int nr_timesteps,
            unsigned int nr_channels)
        {
            return 1ULL * nr_baselines * nr_timesteps * nr_channels * sizeof(Visibility<std::complex<float>>);
        }

        uint64_t sizeof_uvw(
            unsigned int nr_baselines,
            unsigned int nr_timesteps)
        {
            return 1ULL * nr_baselines * nr_timesteps * sizeof(UVWCoordinate<float>);
        }

        uint64_t sizeof_subgrids(
            unsigned int nr_subgrids,
            unsigned int subgrid_size,
            uint64_t nr_correlations)
        {
            return 1ULL * nr_subgrids * nr_correlations * subgrid_size * subgrid_size* sizeof(std::complex<float>);
        }

        uint64_t sizeof_metadata(
            unsigned int nr_subgrids)
        {
            return 1ULL * nr_subgrids * sizeof(Metadata);
        }

        uint64_t sizeof_grid(
            unsigned int grid_size,
            uint64_t nr_correlations)
        {
            return 1ULL * nr_correlations * grid_size * grid_size * sizeof(std::complex<float>);
        }

        uint64_t sizeof_wavenumbers(
            unsigned int nr_channels)
        {
            return 1ULL * nr_channels * sizeof(float);
        }

        uint64_t sizeof_aterms(
            unsigned int nr_stations,
            unsigned int nr_timeslots,
            unsigned int subgrid_size,
            uint64_t nr_correlations)
        {
            return 1ULL * nr_stations * nr_timeslots * nr_correlations * subgrid_size * subgrid_size * sizeof(std::complex<float>);
        }

        uint64_t sizeof_spheroidal(
            unsigned int subgrid_size)
        {
            return 1ULL * subgrid_size * subgrid_size * sizeof(float);
        }

        uint64_t sizeof_avg_aterm_correction(
            unsigned int subgrid_size,
            uint64_t nr_correlations)
        {
            return 1ULL * subgrid_size * subgrid_size * nr_correlations * nr_correlations * sizeof(std::complex<float>);
        }

        uint64_t sizeof_baselines(
            unsigned int nr_baselines)
        {
            return 1ULL * 2 * nr_baselines * sizeof(unsigned int);
        }

        uint64_t sizeof_aterms_offsets(
            unsigned int nr_timeslots)
        {
            return 1ULL * (nr_timeslots + 1) * sizeof(unsigned int);
        }

        /*
            Performance reporting
         */
        #define FW1 12
        #define FW2 8

        void report(
            string name,
            double runtime)
        {
            clog << setw(FW1) << left << string(name) + ": "
                 << setw(FW2) << right << scientific << setprecision(4)
                 << runtime << " s" << endl;
        }

        void report(
            string name,
            double runtime,
            double joules,
            uint64_t flops,
            uint64_t bytes,
            bool ignore_short)
        {
            // Ignore very short measurements
            if (ignore_short && runtime < 1e-3) {
                return;
            }

            double watt = joules / runtime;
            #pragma omp critical (clog)
            {
                clog << setw(FW1) << left << string(name) + ": "
                     << setw(FW2) << right << scientific << setprecision(4)
                     << runtime << " s";
                #if defined(REPORT_OPS)
                if (flops != 0) {
                    clog << ", ";
                    double gops = (flops / runtime) * 1e-9;
                        clog << setw(FW2) << right << fixed << setprecision(2)
                                          << gops << " GOPS";
                }
                #else
                if (flops != 0) {
                    clog << ", ";
                    double gflops = (flops / runtime) * 1e-9;
                        clog << setw(FW2) << right << fixed << setprecision(2)
                                          << gflops << " GFLOPS";
                }
                #endif
                if (bytes != 0) {
                    clog << ", ";
                    clog << setw(FW2) << right << fixed << setprecision(2)
                                      << bytes / runtime * 1e-9 << " GB/s";
                }
                if (watt > 1) {
                    clog << ", ";
                    clog << setw(FW2) << right << fixed << setprecision(2)
                                      << watt << " Watt";
                }
                if (flops != 0 && watt > 1) {
                    clog << ", ";
                    clog << setw(FW2) << right << fixed << setprecision(2)
                                      << (flops / runtime * 1e-9) / watt << " GFLOPS/W";
                }
                if (joules > 1) {
                    clog << ", ";
                    clog << setw(FW2) << right  << fixed << setprecision(2)
                                      << joules << " Joules";
                }

            }
            clog << endl;
        }

        void report(
            string name,
            uint64_t flops,
            uint64_t bytes,
            powersensor::PowerSensor *powerSensor,
            powersensor::State startState,
            powersensor::State endState)
        {
            double seconds = powerSensor->seconds(startState, endState);
            double joules  = powerSensor->Joules(startState, endState);
            report(name, seconds, flops, bytes, joules);
            return;
       }

        void report_visibilities(
            string name,
            double runtime,
            uint64_t nr_visibilities)
        {
            clog << setw(FW1) << left << string(name) + ": "
                 << fixed << setprecision(2)
                 << 1e-6 * nr_visibilities / runtime
                 << " Mvisibilities/s" << endl;
        }

        /*
            Misc
        */
        std::vector<int> split_int(const char *string, const char *delimiter) {
            std::vector<int> splits;
            char * string_buffer = new char [strlen(string)+1];
            std::strcpy (string_buffer, string);
            char *token = strtok(string_buffer, delimiter);
            if (token) splits.push_back(atoi(token));
            while (token) {
                token = strtok(NULL, delimiter);
                if (token) splits.push_back(atoi(token));
            }
            delete [] string_buffer;
            return splits;
        }

        std::vector<std::string> split_string(char *string, const char *delimiter) {
            std::vector<std::string> splits;
            if (!string) {
                return splits;
            }
            char *token = strtok(string, delimiter);
            if (token) splits.push_back(token);
            while (token) {
                token = strtok(NULL, delimiter);
                if (token) splits.push_back(token);
            }
            return splits;
        }

        std::string get_lib_dir() {
            return std::string(IDG_INSTALL_DIR) + "/lib";
        }
    } // namespace auxiliary
} // namespace idg
