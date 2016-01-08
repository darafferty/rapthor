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
#include "KernelsJetson.h"

namespace idg {
    namespace proxy {
        namespace cuda {
            class Jetson : public CUDA {
            public:
                /// Constructors
                Jetson(Parameters params,
                        unsigned deviceNumber = 0,
                        Compiler compiler = default_compiler(),
                        Compilerflags flags = default_compiler_flags(),
                        ProxyInfo info = default_info());

                /// Destructor
                ~Jetson() = default;

                static ProxyInfo default_info();
                static ProxyInfo default_proxyinfo(std::string srcdir, std::string tmpdir);

            public:
                void transform(DomainAtoDomainB direction,
                                    std::complex<float>* grid) override;

                void grid_visibilities(
                    const std::complex<float> *visibilities,
                    const float *uvw,
                    const float *wavenumbers,
                    const int *baselines,
                    std::complex<float> *grid,
                    const float w_offset,
                    const std::complex<float> *aterm,
                    const float *spheroidal) override;

                void degrid_visibilities(
                    std::complex<float> *visibilities,
                    const float *uvw,
                    const float *wavenumbers,
                    const int *baselines,
                    const std::complex<float> *grid,
                    const float w_offset,
                    const std::complex<float> *aterm,
                    const float *spheroidal) override;

            public:
                virtual std::unique_ptr<idg::kernel::cuda::Gridder> get_kernel_gridder() const override;
                virtual std::unique_ptr<idg::kernel::cuda::Degridder> get_kernel_degridder() const override;
                virtual std::unique_ptr<idg::kernel::cuda::GridFFT> get_kernel_fft() const override;
                virtual std::unique_ptr<idg::kernel::cuda::Scaler> get_kernel_scaler() const override;
                virtual std::unique_ptr<idg::kernel::cuda::Adder> get_kernel_adder() const override;
                virtual std::unique_ptr<idg::kernel::cuda::Splitter> get_kernel_splitter() const override;

            }; // class Jetson
        } // namespace cuda
    } // namespace proxy
} // namespace idg
#endif
