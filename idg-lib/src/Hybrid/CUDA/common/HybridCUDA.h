#ifndef IDG_HYBRID_CUDA_H_
#define IDG_HYBRID_CUDA_H_

#include "idg-cpu.h"
#include "idg-cuda.h"

namespace idg {
    namespace proxy {
        namespace hybrid {
            class HybridCUDA : public cuda::CUDA {

                public:
                    HybridCUDA(
                        idg::proxy::cpu::CPU* cpuProxy,
                        CompileConstants constants);

                    ~HybridCUDA();

                private:
                    virtual void do_gridding(
                        const Plan& plan,
                        const float w_offset, // in lambda
                        const float cell_size,
                        const unsigned int kernel_size, // full width in pixels
                        const Array1D<float>& frequencies,
                        const Array3D<Visibility<std::complex<float>>>& visibilities,
                        const Array2D<UVWCoordinate<float>>& uvw,
                        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                        Grid& grid,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array1D<unsigned int>& aterms_offsets,
                        const Array2D<float>& spheroidal) override;

                    virtual void do_degridding(
                        const Plan& plan,
                        const float w_offset, // in lambda
                        const float cell_size,
                        const unsigned int kernel_size, // full width in pixels
                        const Array1D<float>& frequencies,
                        Array3D<Visibility<std::complex<float>>>& visibilities,
                        const Array2D<UVWCoordinate<float>>& uvw,
                        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                        const Grid& grid,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array1D<unsigned int>& aterms_offsets,
                        const Array2D<float>& spheroidal) override;

                    virtual void do_transform(
                        DomainAtoDomainB direction,
                        Array3D<std::complex<float>>& grid) override;

                private:
                    PowerSensor* hostPowerSensor;
                    idg::proxy::cpu::CPU* cpuProxy;

            }; // class HybridCUDA
        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
