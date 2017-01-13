#ifndef IDG_CUDA_GENERIC_H_
#define IDG_CUDA_GENERIC_H_

#include "idg-cuda.h"

/*
    Toggle between two modes of cu::HostMemory allocation
        REDUCE_HOST_MEMORY = 0:
            visibilities and uvw will be completely mapped
            into host memory shared by all threads
            (this takes some time, especially for large buffers)
        REDUCE_HOST_MEMORY = 1:
            every thread allocates private host memory
            to hold data for just one job
            (throughput is lower, due to additional memory copies)
*/
#define REDUCE_HOST_MEMORY 0

namespace cu {
    class HostMemory;
}

namespace idg {
    namespace proxy {
        namespace cuda {
            class Generic : public CUDA {
                public:
                    // Constructor
                    Generic(
                        CompileConstants constants,
                        ProxyInfo info = default_info());

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
                        const Array3D<std::complex<float>>& grid) override;

                private:
                    PowerSensor *hostPowerSensor;

                    #if REDUCE_HOST_MEMORY
                    std::vector<cu::HostMemory*> h_visibilities_;
                    std::vector<cu::HostMemory*> h_uvw_;
                    #else
                    cu::HostMemory *h_visibilities_;
                    cu::HostMemory *h_uvw_;
                    #endif
                    std::vector<cu::HostMemory*> h_grid_;

                    void init_benchmark();
                    bool enable_benchmark = false;
                    int nr_repetitions = 1;
            }; // class Generic

        } // namespace cuda
    } // namespace proxy
} // namespace idg
#endif
