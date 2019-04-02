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
                    virtual bool supports_avg_aterm_correction() {return true;}

                private:
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

                    virtual void initialize(
                        const Plan& plan,
                        const float w_step,
                        const Array1D<float>& shift,
                        const float cell_size,
                        const unsigned int kernel_size,
                        const unsigned int subgrid_size,
                        const Array1D<float>& frequencies,
                        const Array3D<Visibility<std::complex<float>>>& visibilities,
                        const Array2D<UVWCoordinate<float>>& uvw,
                        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                        const Grid& grid,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array1D<unsigned int>& aterms_offsets,
                        const Array2D<float>& spheroidal) override;


                    virtual void run_gridding(
                        const Plan& plan,
                        const float w_step,
                        const Array1D<float>& shift,
                        const float cell_size,
                        const unsigned int kernel_size,
                        const unsigned int subgrid_size,
                        const Array1D<float>& frequencies,
                        const Array3D<Visibility<std::complex<float>>>& visibilities,
                        const Array2D<UVWCoordinate<float>>& uvw,
                        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                        Grid& grid,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array1D<unsigned int>& aterms_offsets,
                        const Array2D<float>& spheroidal) override;


                    virtual void run_degridding(
                        const Plan& plan,
                        const float w_step,
                        const Array1D<float>& shift,
                        const float cell_size,
                        const unsigned int kernel_size,
                        const unsigned int subgrid_size,
                        const Array1D<float>& frequencies,
                        Array3D<Visibility<std::complex<float>>>& visibilities,
                        const Array2D<UVWCoordinate<float>>& uvw,
                        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                        const Grid& grid,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array1D<unsigned int>& aterms_offsets,
                        const Array2D<float>& spheroidal) override;

                    virtual void finish_gridding() override
                    { finish(auxiliary::name_gridding); };

                    virtual void finish_degridding() override
                    { finish(auxiliary::name_degridding); };

                private:
                    void synchronize();
                    void finish(std::string name);

                protected:
                    powersensor::PowerSensor* hostPowerSensor;
                    idg::proxy::cpu::CPU* cpuProxy;

                    /*
                     * Asynchronous (de)gridding state
                     */
                    std::vector<int> jobsize_;
                    std::vector<int> planned_max_nr_subgrids;
                    cu::Stream* hostStream;
                    powersensor::State hostStartState;

            }; // class GenericOptimized

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
