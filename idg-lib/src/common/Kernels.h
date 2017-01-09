#ifndef IDG_KERNELS_H_
#define IDG_KERNELS_H_

#include <cstdint>
#include <cmath>

#include "idg-common.h"


namespace idg {
    namespace kernel {

        class Kernels
        {
            public:
                Kernels(
                    CompileConstants constants,
                    Compiler compiler,
                    Compilerflags flags,
                    ProxyInfo info) :
                    mConstants(constants) {}

                uint64_t flops_gridder(
                    uint64_t nr_channels,
                    uint64_t nr_timesteps,
                    uint64_t nr_subgrids);
                uint64_t bytes_gridder(
                    uint64_t nr_channels,
                    uint64_t nr_timesteps,
                    uint64_t nr_subgrids);
                uint64_t flops_degridder(
                    uint64_t nr_channels,
                    uint64_t nr_timesteps,
                    uint64_t nr_subgrids);
                uint64_t bytes_degridder(
                    uint64_t nr_channels,
                    uint64_t nr_timesteps,
                    uint64_t nr_subgrids);
                uint64_t flops_fft(
                    uint64_t size,
                    uint64_t batch);
                uint64_t bytes_fft(
                    uint64_t size,
                    uint64_t batch);
                uint64_t flops_adder(
                    uint64_t nr_subgrids);
                uint64_t bytes_adder(
                    uint64_t nr_subgrids);
                uint64_t flops_splitter(
                    uint64_t nr_subgrids);
                uint64_t bytes_splitter(
                    uint64_t nr_subgrids);

            protected:
                CompileConstants mConstants;
        };

    } // namespace kernel
} // namespace idg
#endif
