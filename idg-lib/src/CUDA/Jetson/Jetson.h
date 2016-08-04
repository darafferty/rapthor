/**
 *  \class Jetson
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CUDA_JETSON_H_
#define IDG_CUDA_JETSON_H_

#include "../common/CUDA.h"

namespace idg {
    namespace proxy {
        namespace cuda {
            class Jetson : public CUDA {
                public:
                    /// Constructor
                    Jetson(
                        Parameters params,
                        ProxyInfo info = default_info());

                    /// Destructor
                    ~Jetson() = default;

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
            }; // class Jetson

        } // namespace cuda
    } // namespace proxy
} // namespace idg
#endif
