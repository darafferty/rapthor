#include <iostream>
#include <iomanip>
#include <cstdint>
#include <omp.h>

#include "idg-config.h"
#include "auxiliary.h"

using namespace std;

namespace idg {
    namespace auxiliary {

        #define FW1 14
        #define FW2 10

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
                if (flops != 0) {
                    clog << ", ";
                    double gflops = (flops / runtime) * 1e-9;
                        clog << setw(FW2) << right << fixed << setprecision(2)
                                          << gflops << " GFLOPS";
                }
                if (bytes != 0) {
                    clog << ", ";
                    clog << setw(FW2) << right << fixed << setprecision(2)
                                      << bytes / runtime * 1e-9 << " GB/s";
                }
                if  (watt != 0) {
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

        void report_subgrids(
            double runtime,
            uint64_t nr_baselines)
        {
            clog << "throughput: " << 1e-3 * nr_baselines / runtime
                 << " Ksubgrids/s" << endl;
        }

        void report_subgrids(
            const char *name,
            double runtime,
            uint64_t nr_baselines)
        {
            clog << setw(FW1) << left << string(name) + ": "
                 << 1e-3 * nr_baselines / runtime
                 << " Ksubgrids/s" << endl;
        }
    } // namespace auxiliary
} // namespace idg
