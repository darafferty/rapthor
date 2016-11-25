#include <iostream>
#include <iomanip>
#include <cstdint>
#include <omp.h>

#include "idg-config.h"
#include "auxiliary.h"

using namespace std;

namespace idg {
    namespace auxiliary {

        #define FW1 12
        #define FW2 8

        void report(
            const char *name,
            double runtime)
        {
            clog << setw(FW1) << left << string(name) + ": "
                 << setw(FW2) << right << scientific << setprecision(4)
                 << runtime << " s" << endl;
        }

        void report(
            const char *name,
            double runtime,
            uint64_t flops,
            uint64_t bytes,
            double watt)
        {
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
                if (watt != 0) {
                    clog << ", ";
                    clog << setw(FW2) << right << fixed << setprecision(2)
                                      << watt << " Watt";
                }
                if (flops != 0 && watt != 0) {
                    clog << ", ";
                    clog << setw(FW2) << right << fixed << setprecision(2)
                                      << (flops / runtime * 1e-9) / watt << " GFLOPS/W";
                }
            }
            clog << endl;
        }

        void report(
            const char *name,
            uint64_t flops,
            uint64_t bytes,
            PowerSensor *powerSensor,
            PowerSensor::State startState,
            PowerSensor::State endState)
        {
            double seconds = powerSensor->seconds(startState, endState);
            double watts   = powerSensor->Watt(startState, endState);
            double joules  = powerSensor->Joules(startState, endState);
            #pragma omp critical (clog)
            {
                clog << setw(FW1) << left << string(name) + ": "
                     << setw(FW2) << right << scientific << setprecision(4)
                     << seconds << " s";
                #if defined(REPORT_OPS)
                if (flops != 0) {
                    clog << ", ";
                    double gops = (flops / seconds) * 1e-9;
                        clog << setw(FW2) << right << fixed << setprecision(2)
                                          << gops << " GOPS";
                }
                #else
                if (flops != 0) {
                    clog << ", ";
                    double gflops = (flops / seconds) * 1e-9;
                        clog << setw(FW2) << right << fixed << setprecision(2)
                                          << gflops << " GFLOPS";
                }
                #endif
                if (bytes != 0) {
                    clog << ", ";
                    clog << setw(FW2) << right << fixed << setprecision(2)
                                      << bytes / seconds * 1e-9 << " GB/s";
                }
                if (watts != 0) {
                    clog << ", ";
                    clog << setw(FW2) << right << fixed << setprecision(2)
                                      << watts << " Watt";
                }
                if (flops != 0 && watts != 0) {
                    clog << ", ";
                    clog << setw(FW2) << right << fixed << setprecision(2)
                                      << (flops / seconds * 1e-9) / watts << " GFLOPS/W";
                }
                if (joules != 0) {
                    clog << ", ";
                    clog << setw(FW2) << right  << fixed << setprecision(2)
                                      << joules << " Joule";
                }
            }
            clog << endl;
        }

        void report_runtime(double runtime)
        {
            clog << setw(FW1) << left << "runtime: "
                 << setw(FW2) << right << scientific << setprecision(4)
                 << runtime << " s" << endl;
        }

        void report_visibilities(
            double runtime,
            uint64_t nr_baselines,
            uint64_t nr_time,
            uint64_t nr_channels)
        {
            uint64_t nr_visibilities = nr_baselines * nr_time * nr_channels;

            clog << setw(FW1) << left << "throughput: "
                 << setw(FW2) << right << scientific << setprecision(4)
                 << 1e-6 * nr_visibilities / runtime
                 << " Mvisibilities/s" << endl;
        }

        void report_visibilities(
            const char *name,
            double runtime,
            uint64_t nr_baselines,
            uint64_t nr_time,
            uint64_t nr_channels)
        {
            uint64_t nr_visibilities = nr_baselines * nr_time * nr_channels;
            clog << setw(FW1) << left << string(name) + ": "
                 << 1e-6 * nr_visibilities / runtime
                 << " Mvisibilities/s" << endl;
        }


        void report_visibilities(
            const char *name,
            double runtime,
            uint64_t total_nr_time,
            uint64_t nr_channels)
        {
            uint64_t nr_visibilities = total_nr_time * nr_channels;
            clog << setw(FW1) << left << string(name) + ": "
                 << 1e-6 * nr_visibilities / runtime
                 << " Mvisibilities/s" << endl;
        }


        void report_subgrids(
            double runtime,
            uint64_t nr_subgrids)
        {
            clog << "throughput: " << 1e-3 * nr_subgrids / runtime
                 << " Ksubgrids/s" << endl;
        }

        void report_subgrids(
            const char *name,
            double runtime,
            uint64_t nr_subgrids)
        {
            clog << setw(FW1) << left << string(name) + ": "
                 << 1e-3 * nr_subgrids / runtime
                 << " Ksubgrids/s" << endl;
        }

        std::vector<int> split_int(char *string, const char *delimiter) {
            std::vector<int> splits;
            char *token = strtok(string, delimiter);
            if (token) splits.push_back(atoi(token));
            while (token) {
                token = strtok(NULL, delimiter);
                if (token) splits.push_back(atoi(token));
            }
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

    } // namespace auxiliary
} // namespace idg
