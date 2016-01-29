/**
 *  \class Maxwell
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CUDA_MAXWELL_H_
#define IDG_CUDA_MAXWELL_H_

#include "idg-cuda.h"
#include "KernelsMaxwell.h"

namespace idg {
    namespace proxy {
        namespace cuda {
            class Maxwell : public CUDA {
            public:
                /// Constructors
                Maxwell(Parameters params,
                        unsigned deviceNumber = 0,
                        Compiler compiler = default_compiler(),
                        Compilerflags flags = default_compiler_flags(),
                        ProxyInfo info = default_info());

                /// Destructor
                ~Maxwell() = default;

            public:
                static std::string append(Compilerflags flags);

            public:
                virtual std::unique_ptr<idg::kernel::cuda::Gridder> get_kernel_gridder() const override;
                virtual std::unique_ptr<idg::kernel::cuda::Degridder> get_kernel_degridder() const override;
                virtual std::unique_ptr<idg::kernel::cuda::GridFFT> get_kernel_fft() const override;
                virtual std::unique_ptr<idg::kernel::cuda::Scaler> get_kernel_scaler() const override;
                virtual std::unique_ptr<idg::kernel::cuda::Adder> get_kernel_adder() const override;
                virtual std::unique_ptr<idg::kernel::cuda::Splitter> get_kernel_splitter() const override;

            }; // class Maxwell

        } // namespace cuda
    } // namespace proxy
} // namespace idg
#endif
