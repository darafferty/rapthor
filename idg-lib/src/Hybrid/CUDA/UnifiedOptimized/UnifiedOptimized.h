#ifndef IDG_HYBRID_UNIFIED_OPTIMIZED_H_
#define IDG_HYBRID_UNIFIED_OPTIMIZED_H_

#include "idg-hybrid-cuda.h"

namespace idg {
    namespace proxy {
        namespace hybrid {

            class UnifiedOptimized : public cuda::CUDA {
                public:
                    UnifiedOptimized(
                        ProxyInfo info = default_info());

                    ~UnifiedOptimized();

                    virtual bool supports_wstack_gridding() { return false; }
                    virtual bool supports_wstack_degridding() { return false; }

                    virtual void do_gridding(
                        const Plan& plan,
                        const float w_step, // in lambda
                        const Array1D<float>& shift,
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
                        const Array1D<float>& shift,
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

                    Grid get_grid(
                        size_t nr_w_layers,
                        size_t nr_correlations,
                        size_t height,
                        size_t width);

                private:
                    powersensor::PowerSensor *hostPowerSensor;
                    idg::proxy::cpu::CPU* cpuProxy;

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
                        const int grid_size,
                        void *visibilities,
                        void *uvw);

                    void* allocate_memory(size_t bytes);
                    void free_memory(void *ptr);
                    void free_memory();
                    std::vector<cu::UnifiedMemory*> memory;

            }; // class UnifiedOptimized

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
