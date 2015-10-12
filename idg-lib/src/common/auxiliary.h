#ifndef IDG_AUX_H_
#define IDG_AUX_H_

#include <cstdint>

namespace idg {
  namespace auxiliary {

    void report(const char *name,
         		double runtime,
         		uint64_t flops,
         		uint64_t bytes);

    void report_runtime(double runtime);

    void report_visibilities(double runtime,
			                 uint64_t nr_baselines,
			                 uint64_t nr_time,
			                 uint64_t nr_channels);

    void report_subgrids(double runtime,
			             uint64_t nr_baselines);

    void report_power(double runtime,
                      double watt,
                      double joules);

  } // namespace auxiliary
} // namespace idg

#endif
