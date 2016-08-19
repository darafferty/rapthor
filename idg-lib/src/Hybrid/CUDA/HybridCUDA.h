/**
 *  \class HybridCUDA
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_HYBRID_CUDA_H_
#define IDG_HYBRID_CUDA_H_

#include "idg-cpu.h"
#include "idg-cuda.h"

namespace idg {
    namespace proxy {
        namespace hybrid {
            class HybridCUDA : public Proxy {

                public:
                    /// Constructors
                    HybridCUDA(Parameters params);

                    /// Destructor
                    ~HybridCUDA();

                    /// Assignment
                    HybridCUDA& operator=(const HybridCUDA& rhs) = delete;

                /*
                    High level routines
                    These routines operate on grids
                */
                virtual void grid_visibilities(
                    const std::complex<float> *visibilities,
                    const float *uvw,
                    const float *wavenumbers,
                    const int *baselines,
                    std::complex<float> *grid,
                    const float w_offset,
                    const int kernel_size,
                    const std::complex<float> *aterm,
                    const int *aterm_offsets,
                    const float *spheroidal) override;

                virtual void degrid_visibilities(
                    std::complex<float> *visibilities,
                    const float *uvw,
                    const float *wavenumbers,
                    const int *baselines,
                    const std::complex<float> *grid,
                    const float w_offset,
                    const int kernel_size,
                    const std::complex<float> *aterm,
                    const int *aterm_offsets,
                    const float *spheroidal) override;

                virtual void transform(DomainAtoDomainB direction,
                    std::complex<float>* grid) override;

                private:
                    idg::proxy::cpu::Optimized cpu;
                    idg::proxy::cuda::Generic cuda;

                    void init_benchmark();
                    bool enable_benchmark = false;
                    int nr_repetitions = 1;
            }; // class HybridCUDA
        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
