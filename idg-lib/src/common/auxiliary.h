#ifndef IDG_AUX_H_
#define IDG_AUX_H_

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "idg-config.h"
#include "idg-powersensor.h"

inline int min(int a, int b) {
    return a < b ? a : b;
}

inline int max(int a, int b) {
    return a > b ? a : b;
}

namespace idg {
    namespace auxiliary {

        void report(
            const char *name,
            double runtime);

        void report(
            const char *name,
            double runtime,
            uint64_t flops,
            uint64_t bytes,
            double watt=0);

        void report(
            const char *name,
            uint64_t flops,
            uint64_t bytes,
            PowerSensor *powerSensor,
            PowerSensor::State startState,
            PowerSensor::State endState);

        void report_runtime(
            double runtime);

        void report_visibilities(
            const char *name,
            double runtime,
    		uint64_t nr_visibilities);

        void report_subgrids(
            const char *name,
            double runtime,
            uint64_t nr_subgrids);

        void report_subgrids(
            double runtime,
            uint64_t nr_subgrids);

        std::vector<int> split_int(char *string, const char *delimiter);
        std::vector<std::string> split_string(char *string, const char *delimiter);

        std::string get_lib_dir();
    } // namespace auxiliary
} // namespace idg

#endif
