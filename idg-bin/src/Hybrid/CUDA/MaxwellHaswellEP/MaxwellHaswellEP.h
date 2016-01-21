/**
 *  \class MaxwellHaswellEP
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_HYBRID_CUDA_MAXWELLHASWELLEP_H_
#define IDG_HYBRID_CUDA_MAXWELLHASWELLEP_H_

#include <dlfcn.h>
#include <cuda.h>
#include <cudaProfiler.h>

#include "idg-hybrid-cuda.h"

namespace idg {
    namespace proxy {
        namespace hybrid {
            class MaxwellHaswellEP : public Proxy {

                public:
                    /// Constructors
                    MaxwellHaswellEP(Parameters params);

                    /// Destructor
                    ~MaxwellHaswellEP();

                    /// Assignment
                    MaxwellHaswellEP& operator=(const MaxwellHaswellEP& rhs) = delete;

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
                    idg::proxy::cpu::HaswellEP cpu;
                    idg::proxy::cuda::Maxwell cuda;

            }; // class MaxwellHaswellEP
        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
