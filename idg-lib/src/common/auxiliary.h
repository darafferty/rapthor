#ifndef IDG_AUX_H_
#define IDG_AUX_H_

#include <cstdint>

#include "idg-config.h"

namespace idg {
    namespace auxiliary {

        void report(
            const char *name,
            double runtime,
            uint64_t flops,
            uint64_t bytes,
            double watt=0);

        void report_runtime(
            double runtime);

        void report_visibilities(
            const char *name,
            double runtime,
    		uint64_t nr_baselines,
    		uint64_t nr_time,
    		uint64_t nr_channels);

        void report_visibilities(
            double runtime,
    		uint64_t nr_baselines,
    		uint64_t nr_time,
    		uint64_t nr_channels);

        void report_subgrids(
            const char *name,
            double runtime,
            uint64_t nr_subgrids);

        void report_subgrids(
            double runtime,
            uint64_t nr_subgrids);

    } // namespace auxiliary
} // namespace idg

#endif
