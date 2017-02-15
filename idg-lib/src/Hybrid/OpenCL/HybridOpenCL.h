/**
 *  \class OpenCL
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_HYBRID_OPENCL_REFERENCE_H_
#define IDG_HYBRID_OPENCL_REFERENCE_H_

#include "idg-hybrid-opencl.h"

namespace idg {
    namespace proxy {
        namespace hybrid {
            class HybridOpenCL : public Proxy {

                public:
                    /// Constructors
                    HybridOpenCL(Parameters params);

                    /// Destructor
                    ~HybridOpenCL();

                    /// Assignment
                    HybridOpenCL& operator=(const HybridOpenCL& rhs) = delete;

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
                    const float w_step,
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
                    const float w_step,
                    const int kernel_size,
                    const std::complex<float> *aterm,
                    const int *aterm_offsets,
                    const float *spheroidal) override;

                virtual void transform(DomainAtoDomainB direction,
                    std::complex<float>* grid) override;

                private:
                    idg::proxy::cpu::Optimized cpu;
                    idg::proxy::opencl::Generic opencl;

            }; // class HybridOpenCL
        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
