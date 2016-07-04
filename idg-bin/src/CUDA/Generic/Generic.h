/**
 *  \class Generic
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CUDA_GENERIC_H_
#define IDG_CUDA_GENERIC_H_

#include "idg-cuda.h"

namespace idg {
    namespace proxy {
        namespace cuda {
            class Generic : public CUDA {
                public:
                    /// Constructor
                    Generic(
                        Parameters params,
                        ProxyInfo info = default_info());

                    /// Destructor
                    ~Generic() = default;

                public:
                    // High level interface, inherited from Proxy
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

            }; // class Generic

        } // namespace cuda
    } // namespace proxy
} // namespace idg
#endif
