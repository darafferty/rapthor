#ifndef IDG_CPU_H_
#define IDG_CPU_H_

#include "idg-common.h"

#include "InstanceCPU.h"

namespace idg {
    namespace proxy {
        namespace cpu {

            class CPU : public Proxy
            {
                public:
                    // Constructor
                    CPU(
                        std::vector<std::string> libraries);

                    // Disallow assignment and pass-by-value
                    CPU& operator=(const CPU& rhs) = delete;
                    CPU(const CPU& v) = delete;

                    // Destructor
                    virtual ~CPU();

                    virtual bool supports_wstack_gridding() {return kernels.has_adder_wstack();}
                    virtual bool supports_wstack_degridding() {return kernels.has_splitter_wstack();}
                    virtual bool supports_avg_aterm_correction() {return true;}

                    kernel::cpu::InstanceCPU& get_kernels() { return kernels; }

                private:
                    // Routines
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

                    virtual void do_transform(
                        DomainAtoDomainB direction,
                        Array3D<std::complex<float>>& grid) override;

                protected:
                    kernel::cpu::InstanceCPU kernels;
                    powersensor::PowerSensor *powerSensor;

                    struct {
                        std::vector<std::unique_ptr<Plan>> plans;
                        float w_step; // in lambda
                        Array1D<float> shift;
                        float cell_size;
                        float image_size;
                        unsigned int kernel_size;
                        unsigned int grid_size;
                        unsigned int subgrid_size;
                        Array1D<float> wavenumbers;
                        Array4D<Visibility<std::complex<float>>> visibilities;
                        Array3D<UVWCoordinate<float>> uvw;
                        Array2D<std::pair<unsigned int,unsigned int>> baselines;
                        std::vector<Array4D<std::complex<float>>> subgrids;
                    } m_calibrate_state;


            }; // end class CPU

        } // end namespace cpu
    } // end namespace proxy
} // end namespace idg
#endif
