/**
 *  \class Generic
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_OPENCL_GENERIC_H_
#define IDG_OPENCL_GENERIC_H_

#include "idg-opencl.h"

namespace idg {
    namespace proxy {
        namespace opencl {
            class Generic : public OpenCLNew {
                public:
                    /// Constructor
                    Generic(
                        Parameters params);

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

                private:
                    //#if REDUCE_HOST_MEMORY
                    //std::vector<cl::Buffer*> h_visibilities_;
                    //std::vector<cl::Buffer*> h_uvw_;
                    //#else
                    //cu::Buffer *h_visibilities;
                    //cu::Buffer *h_uvw;
                    //#endif
                    //std::vector<cu::HostMemory*> h_grid_;
            }; // class Generic

        } // namespace opencl
    } // namespace proxy
} // namespace idg
#endif
