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
                    
                    kernel::cpu::InstanceCPU& get_kernels() { return kernels; }

                private:
                    // Routines
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

                    void grid_onto_subgrids(
                        const Plan& plan,
                        const float w_step,
                        const unsigned int grid_size,
                        const float image_size,
                        const Array1D<float>& wavenumbers,
                        const Array3D<Visibility<std::complex<float>>>& visibilities,
                        const Array2D<UVWCoordinate<float>>& uvw,
                        const Array2D<float>& spheroidal,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        Array4D<std::complex<float>>& subgrids);

                    virtual void add_subgrids_to_grid(
                        const Plan& plan,
                        const float w_step,
                        const Array4D<std::complex<float>>& subgrids,
                        Grid& grid);

                    virtual void split_grid_into_subgrids(
                        const Plan& plan,
                        const float w_step,
                        Array4D<std::complex<float>>& subgrids,
                        const Grid& grid);

                    virtual void degrid_from_subgrids(
                        const Plan& plan,
                        const float w_step,
                        const unsigned int grid_size,
                        const float image_size,
                        const Array1D<float>& wavenumbers,
                        Array3D<Visibility<std::complex<float>>>& visibilities,
                        const Array2D<UVWCoordinate<float>>& uvw,
                        const Array2D<float>& spheroidal,
                        const Array4D<Matrix2x2<std::complex<float>>>& aterms,
                        const Array4D<std::complex<float>>& subgrids);


                protected:
                    kernel::cpu::InstanceCPU kernels;
                    powersensor::PowerSensor *powerSensor;

            }; // end class CPU

        } // end namespace cpu
    } // end namespace proxy
} // end namespace idg
#endif
