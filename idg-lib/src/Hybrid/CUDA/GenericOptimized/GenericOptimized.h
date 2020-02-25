#ifndef IDG_HYBRID_GENERIC_OPTIMIZED_H_
#define IDG_HYBRID_GENERIC_OPTIMIZED_H_

#include <thread>

#include "idg-cpu.h"
#include "CUDA/common/CUDA.h"

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
                        const Array2D<UVW<float>>& uvw,
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
                        const Array2D<UVW<float>>& uvw,
                        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                        const Grid& grid,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array1D<unsigned int>& aterms_offsets,
                        const Array2D<float>& spheroidal) override;

                    virtual void do_transform(
                        DomainAtoDomainB direction,
                        Array3D<std::complex<float>>& grid) override;

                    void run_gridding(
                        const Plan& plan,
                        const float w_step,
                        const Array1D<float>& shift,
                        const float cell_size,
                        const unsigned int kernel_size,
                        const unsigned int subgrid_size,
                        const Array1D<float>& frequencies,
                        const Array3D<Visibility<std::complex<float>>>& visibilities,
                        const Array2D<UVW<float>>& uvw,
                        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                        Grid& grid,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array1D<unsigned int>& aterms_offsets,
                        const Array2D<float>& spheroidal);

                    void run_degridding(
                        const Plan& plan,
                        const float w_step,
                        const Array1D<float>& shift,
                        const float cell_size,
                        const unsigned int kernel_size,
                        const unsigned int subgrid_size,
                        const Array1D<float>& frequencies,
                        Array3D<Visibility<std::complex<float>>>& visibilities,
                        const Array2D<UVW<float>>& uvw,
                        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                        const Grid& grid,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array1D<unsigned int>& aterms_offsets,
                        const Array2D<float>& spheroidal);

                    virtual void do_calibrate_init(
                        std::vector<std::unique_ptr<Plan>> &&plans,
                        float w_step, // in lambda
                        Array1D<float> &&shift,
                        float cell_size,
                        unsigned int kernel_size, // full width in pixels
                        unsigned int subgrid_size,
                        const Array1D<float> &frequencies,
                        Array4D<Visibility<std::complex<float>>> &&visibilities,
                        Array4D<Visibility<float>> &&weights,
                        Array3D<UVW<float>> &&uvw,
                        Array2D<std::pair<unsigned int,unsigned int>> &&baselines,
                        const Grid& grid,
                        const Array2D<float>& spheroidal) override;

                    virtual void do_calibrate_update(
                        const int station_nr,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
                        Array3D<double>& hessian,
                        Array2D<double>& gradient,
                        double &residual) override;

                    virtual void do_calibrate_finish() override;

                    virtual void do_calibrate_init_hessian_vector_product() override;

                    virtual void do_calibrate_update_hessian_vector_product1(
                        const int station_nr,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
                        const Array2D<float>& parameter_vector) override;

                    virtual void do_calibrate_update_hessian_vector_product2(
                        const int station_nr,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array4D<Matrix2x2<std::complex<float>>>& derivative_aterms,
                        Array2D<float>& parameter_vector) override;

                    virtual Plan* make_plan(
                        const int kernel_size,
                        const int subgrid_size,
                        const int grid_size,
                        const float cell_size,
                        const Array1D<float>& frequencies,
                        const Array2D<UVW<float>>& uvw,
                        const Array1D<std::pair<unsigned int,unsigned int>>& baselines,
                        const Array1D<unsigned int>& aterms_offsets,
                        Plan::Options options);

                protected:
                    powersensor::PowerSensor* hostPowerSensor;
                    idg::proxy::cpu::CPU* cpuProxy;

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
                        unsigned int nr_baselines;
                        unsigned int nr_timesteps;
                        unsigned int nr_channels;
                        Array3D<UVW<float>> uvw;
                        std::vector<unsigned int> d_sums_ids;
                        unsigned int d_lmnp_id;
                        std::vector<unsigned int> d_metadata_ids;
                        std::vector<unsigned int> d_subgrids_ids;
                        std::vector<unsigned int> d_visibilities_ids;
                        std::vector<unsigned int> d_weights_ids;
                        std::vector<unsigned int> d_uvw_ids;
                        std::vector<unsigned int> d_aterm_idx_ids;
                        Array3D<Visibility<std::complex<float>>> hessian_vector_product_visibilities;
                    } m_calibrate_state;

                    // Note:
                    //     kernel_calibrate internally assumes max_nr_terms = 8
                    //     and will process larger values of nr_terms in batches
                    const unsigned int m_calibrate_max_nr_terms = 8;

            }; // class GenericOptimized

        } // namespace hybrid
    } // namespace proxy
} // namespace idg

#endif
