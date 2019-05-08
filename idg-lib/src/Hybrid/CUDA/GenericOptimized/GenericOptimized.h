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

                    virtual void do_calibrate_init(
                        std::vector<std::unique_ptr<Plan>> &&plans,
                        float w_step, // in lambda
                        Array1D<float> &&shift,
                        float cell_size,
                        unsigned int kernel_size, // full width in pixels
                        unsigned int subgrid_size,
                        const Array1D<float> &frequencies,
                        Array4D<Visibility<std::complex<float>>> &&visibilities,
                        Array3D<UVWCoordinate<float>> &&uvw,
                        Array2D<std::pair<unsigned int,unsigned int>> &&baselines,
                        const Grid& grid,
                        const Array2D<float>& spheroidal) override;

                    virtual void do_calibrate_update(
                        const int station_nr,
                        const Array3D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array3D<Matrix2x2<std::complex<float>>>& derivative_aterms,
                        Array2D<std::complex<float>>& hessian,
                        Array1D<std::complex<float>>& gradient) override;

                    virtual void do_calibrate_finish() override;

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

                    /*
                     * Calibration state
                     */
                    struct {
                        std::vector<std::unique_ptr<Plan>> plans;
                        float w_step; // in lambda
                        Array1D<float> shift;
                        float cell_size;
                        float image_size;
                        unsigned int kernel_size;
                        long unsigned int grid_size;
                        unsigned int subgrid_size;
                        Array1D<float> wavenumbers;
                        Array4D<Visibility<std::complex<float>>> visibilities;
                        Array3D<UVWCoordinate<float>> uvw;
                        Array2D<std::pair<unsigned int,unsigned int>> baselines;
                        std::vector<Array4D<std::complex<float>>> subgrids;
                        unsigned int d_scratch_pix_id;
                        unsigned int d_scratch_sum_id;
                        unsigned int d_hessian_id;
                        unsigned int d_gradient_id;
                        unsigned int d_aterms_deriv_id;
                    } m_calibrate_state;

                    // Note: kernel_calibrate processes nr_terms+1 terms
                    //       and internally assumes max_nr_terms = 8
                    const unsigned int m_calibrate_max_nr_terms = 8;

            }; // class GenericOptimized

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
