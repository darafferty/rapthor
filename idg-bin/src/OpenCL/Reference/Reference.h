/**
 *  \class Reference
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_OPENCL_REFERENCE_H_
#define IDG_OPENCL_REFERENCE_H_

#include "idg-opencl.h"

namespace idg {
    namespace proxy {
        namespace opencl {

            class Reference : public OpenCL {

            public:
                /// Constructors
                Reference(
                    Parameters params,
                    unsigned deviceNumber = 0);

                /// Destructor
                ~Reference() = default;

            public:
                virtual std::unique_ptr<idg::kernel::opencl::Gridder> get_kernel_gridder() const override;
                virtual std::unique_ptr<idg::kernel::opencl::Degridder> get_kernel_degridder() const override;
                virtual std::unique_ptr<idg::kernel::opencl::GridFFT> get_kernel_fft() const override;
                virtual std::unique_ptr<idg::kernel::opencl::Scaler> get_kernel_scaler() const override;
                virtual std::unique_ptr<idg::kernel::opencl::Adder> get_kernel_adder() const override;
                virtual std::unique_ptr<idg::kernel::opencl::Splitter> get_kernel_splitter() const override;

            }; // class Reference

        } // namespace opencl
    } // namespace proxy
} // namespace idg

#endif
