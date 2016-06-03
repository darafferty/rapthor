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

#include <complex>

#include "idg-cuda.h"

namespace idg {
    namespace proxy {
        namespace cuda {
            class Jetson : public CUDA {
                public:
                    /// Constructor
                    Jetson(
                        Parameters params,
                        unsigned deviceNumber = 0,
                        Compiler compiler = default_compiler(),
                        Compilerflags flags = default_compiler_flags(),
                        ProxyInfo info = default_info());

                    /// Destructor
                    ~Jetson() = default;

                protected:
                    virtual dim3 get_block_gridder() const override;
                    virtual dim3 get_block_degridder() const override;
                    virtual dim3 get_block_adder() const override;
                    virtual dim3 get_block_splitter() const override;
                    virtual dim3 get_block_scaler() const override;
                    virtual int get_gridder_batch_size() const override;
                    virtual int get_degridder_batch_size() const override;
                    virtual std::string append(Compilerflags flags) const override;

                public:
                    /*
                        High level routines
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
            }; // class Jetson

        } // namespace cuda
    } // namespace proxy
} // namespace idg
#endif
