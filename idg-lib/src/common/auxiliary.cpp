#include <iostream>
#include <iomanip>
#include <cstdint>
#include <omp.h>

#include "idg-config.h"
#include "auxiliary.h"

using namespace std;

namespace idg {
    namespace auxiliary {


        void report(const char *name,
              double runtime,
              uint64_t flops,
              uint64_t bytes,
              double watt)
        {
            const int fw1 = 12;
            const int fw2 = 10;
            const int fw3 = 10;

            string name_str(name);


            #pragma omp critical (clog)
            {
            clog << setw(fw1) << left << name_str + ": "
                 << setw(fw2) << right << scientific << setprecision(4)
                 << runtime << " s";
            if (flops != 0)
                clog << ", ";
                clog << setw(fw2) << right << fixed << setprecision(2)
                                  << flops / runtime * 1e-9 << " GFLOPS";
            if (bytes != 0)
                clog << ", ";
                clog << setw(fw3) << right << fixed << setprecision(2)
                                  << bytes / runtime * 1e-9 << " GB/s";
            if  (watt != 0)
                clog << ", ";
                clog << setw(fw3) << right << fixed << setprecision(2)
                                  << watt << " Watt";
            }
            clog << endl;
        }

        void report_runtime(double runtime)
        {
            const int fw1 = 12;
            const int fw2 = 10;

            clog << setw(fw1) << left << "runtime: "
                 << setw(fw2) << right << scientific << setprecision(4)
                 << runtime << " s" << endl;
        }

        void report_visibilities(double runtime,
                       uint64_t nr_baselines,
                       uint64_t nr_time,
                       uint64_t nr_channels)
        {
            const int fw1 = 12;
            const int fw2 = 10;

            uint64_t nr_visibilities = nr_baselines * nr_time * nr_channels;

            clog << setw(fw1) << left << "throughput: "
                 << setw(fw2) << right << scientific << setprecision(4)
                 << 1e-6 * nr_visibilities / runtime
                 << " Mvisibilities/s" << endl;
        }

        void report_subgrids(double runtime,
                   uint64_t nr_baselines)
        {
            clog << "throughput: " << 1e-3 * nr_baselines / runtime
                 << " Ksubgrids/s" << endl;
        }

        void report_power(double runtime,
                          double watt,
                          double joules)
        {
            clog << fixed
                 << "   runtime: " << runtime << "s, "
                 << watt << " W, energy: "
                 << joules << " J" << endl;
        }
    } // namespace auxiliary
} // namespace idg
