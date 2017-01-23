#ifndef IDG_OPENCL_GENERIC_H_
#define IDG_OPENCL_GENERIC_H_

#include "idg-opencl.h"

namespace idg {
    namespace proxy {
        namespace opencl {
            class Generic : public OpenCL {
                public:
                    // Constructor
                    Generic(
                        CompileConstants constants);

                    // Destructor
                    ~Generic();

                public:
                    virtual void gridding(
                        const Plan& plan,
                        const float w_offset, // in lambda
                        const float cell_size,
                        const unsigned int kernel_size, // full width in pixels
                        const Array1D<float>& frequencies,
                        const Array3D<Visibility<std::complex<float>>>& visibilities,
                        const Array2D<UVWCoordinate<float>>& uvw,
                        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                        Array3D<std::complex<float>>& grid,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array1D<unsigned int>& aterms_offsets,
                        const Array2D<float>& spheroidal) override;

                    using Proxy::gridding;

                    virtual void degridding(
                        const Plan& plan,
                        const float w_offset, // in lambda
                        const float cell_size,
                        const unsigned int kernel_size, // full width in pixels
                        const Array1D<float>& frequencies,
                        Array3D<Visibility<std::complex<float>>>& visibilities,
                        const Array2D<UVWCoordinate<float>>& uvw,
                        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                        const Array3D<std::complex<float>>& grid,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array1D<unsigned int>& aterms_offsets,
                        const Array2D<float>& spheroidal) override;

                    using Proxy::degridding;

                    virtual void transform(
                        DomainAtoDomainB direction,
                        Array3D<std::complex<float>>& grid) override;

                private:
                    PowerSensor *hostPowerSensor;

            }; // class Generic

        } // namespace opencl
    } // namespace proxy
} // namespace idg
#endif
