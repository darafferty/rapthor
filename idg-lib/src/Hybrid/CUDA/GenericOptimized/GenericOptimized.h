#ifndef IDG_HYBRID_GENERIC_OPTIMIZED_H_
#define IDG_HYBRID_GENERIC_OPTIMIZED_H_

#include "idg-hybrid-cuda.h"

namespace cu {
    class Stream;
    class Event;
}

namespace idg {
    namespace proxy {
        namespace hybrid {

            class GenericOptimized : public cuda::CUDA {

                public:
                    GenericOptimized();
                    ~GenericOptimized();

                    virtual bool supports_wstack_gridding()   { return cpuProxy->supports_wstack_gridding(); }
                    virtual bool supports_wstack_degridding() { return cpuProxy->supports_wstack_degridding(); }

                private:
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

                    void initialize_gridding(
                        const Plan& plan,
                        const float cell_size,
                        const unsigned int kernel_size,
                        const unsigned int subgrid_size,
                        const unsigned int grid_size,
                        const Array1D<float>& frequencies,
                        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array2D<float>& spheroidal);

                    void run_gridding(
                        const Plan& plan,
                        const float w_step,
                        const float cell_size,
                        const unsigned int subgrid_size,
                        const unsigned int nr_stations,
                        const Array1D<float>& wavenumbers,
                        const Array3D<Visibility<std::complex<float>>>& visibilities,
                        const Array2D<UVWCoordinate<float>>& uvw,
                        Grid& grid);

                    void finish_gridding();

                protected:
                    powersensor::PowerSensor* hostPowerSensor;
                    idg::proxy::cpu::CPU* cpuProxy;
                    cu::Stream* hostStream;

                    /*
                     * Asynchronous gridding state
                     */
                    std::vector<cu::Event*> inputFree;
                    std::vector<cu::Event*> inputReady;
                    std::vector<cu::Event*> outputFree;
                    std::vector<cu::Event*> outputReady;
                    std::vector<cu::Event*> adderFinished;
                    int global_id = 0;
                    std::vector<int> planned_max_nr_subgrids;
                    powersensor::State hostStartState;

            }; // class GenericOptimized

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
