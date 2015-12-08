/**
 *  \class Kepler
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CUDA_KEPLER_H_
#define IDG_CUDA_KEPLER_H_

#include "CUDA.h"
#include "KernelsKepler.h"

namespace idg {
    namespace proxy {
        namespace cuda {

            class Kepler : public CUDA {
            public:
                /// Constructors
                Kepler(Parameters params,
                       unsigned deviceNumber = 0,
                       Compiler compiler = default_compiler(),
                       Compilerflags flags = default_compiler_flags(),
                       ProxyInfo info = default_info());

                /// Destructor
                ~Kepler() = default;

                static ProxyInfo default_info();
                static ProxyInfo default_proxyinfo(std::string srcdir, std::string tmpdir);

            public:
                virtual std::unique_ptr<idg::kernel::cuda::Gridder> get_kernel_gridder() const override;
                virtual std::unique_ptr<idg::kernel::cuda::Degridder> get_kernel_degridder() const override;
                virtual std::unique_ptr<idg::kernel::cuda::GridFFT> get_kernel_fft() const override;
                // TODO: add Adder and Splitter

            }; // class Kepler

        } // namespace cuda
    } // namespace proxy
} // namespace idg
#endif
