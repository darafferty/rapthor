#ifndef IDG_HYBRID_CUDA_H_
#define IDG_HYBRID_CUDA_H_

#include "idg-cpu.h"
#include "idg-cuda.h"

namespace idg {
    namespace proxy {
        namespace hybrid {
#if 0
            class HybridCUDA : public cuda::CUDA {

                public:
                    HybridCUDA(
                        idg::proxy::cpu::CPU* cpuProxy);

                    ~HybridCUDA();

                    virtual bool supports_wstack_gridding() {return cpuProxy->supports_wstack_gridding();}
                    virtual bool supports_wstack_degridding() {return cpuProxy->supports_wstack_degridding();}

                private:
                    void initialize_memory(
                        const Plan& plan,
                        const std::vector<int> jobsize,
                        const int nr_streams,
                        const int nr_baselines,
                        const int nr_timesteps,
                        const int nr_channels,
                        const int nr_stations,
                        const int nr_timeslots,
                        const int subgrid_size,
                        void *visibilities,
                        void *uvw);

                    virtual void do_gridding(
                        const Plan& plan,
                        const float w_step, // in lambda
                        const float cell_size,
                        const unsigned int kernel_size, // full width in pixels
                        const unsigned int subgrid_size,
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
                        const float w_step, // in lambda
                        const float cell_size,
                        const unsigned int kernel_size, // full width in pixels
                        const unsigned int subgrid_size,
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

                protected:
                    powersensor::PowerSensor* hostPowerSensor;
                    idg::proxy::cpu::CPU* cpuProxy;

            }; // class HybridCUDA
#endif
        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
