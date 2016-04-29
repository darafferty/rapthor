#ifndef IDG_KERNELS_H_
#define IDG_KERNELS_H_

#include <cstdint>
#include <cmath>

#include "Parameters.h"

//#define COUNT_SINCOS_AS_FLOPS
#if defined(COUNT_SINCOS_AS_FLOPS)
#define FLOPS_PER_SINCOS 8
#endif

namespace idg {
namespace kernel {

    /*
        Flop and byte count
    */
    uint64_t flops_gridder(Parameters &parameters, int nr_baselines, int nr_subgrids);
    uint64_t bytes_gridder(Parameters &parameters, int nr_baselines, int nr_subgrids);
    uint64_t flops_degridder(Parameters &parameters, int nr_baselines, int nr_subgrids);
    uint64_t bytes_degridder(Parameters &parameters, int nr_baselines, int nr_subgrids);
    uint64_t flops_fft(Parameters &parameters, int size, int batch);
    uint64_t bytes_fft(Parameters &parameters, int size, int batch);
    uint64_t flops_adder(Parameters &parameters, int nr_subgrids);
    uint64_t bytes_adder(Parameters &parameters, int nr_subgrids);
    uint64_t flops_splitter(Parameters &parameters, int nr_subgrids);
    uint64_t bytes_splitter(Parameters &parameters, int nr_subgrids);

} // namespace kernel
} // namespace idg
#endif
