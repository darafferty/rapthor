/**
 *  \class Generic
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_OPENCL_GENERIC_H_
#define IDG_OPENCL_GENERIC_H_


#include "idg-opencl.h"


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


namespace idg {
    namespace proxy {
        namespace opencl {
            class Generic : public OpenCL {
                public:
                    // Constructor
                    Generic(
                        Parameters params);

                    // Destructor
                    ~Generic();

                public:
                    // High level interface, inherited from Proxy
                    virtual void grid_visibilities(
                        const std::complex<float> *visibilities,
                        const float *uvw,
                        const float *wavenumbers,
                        const int *baselines,
                        std::complex<float> *grid,
                        const float w_offset,
                        const int kernel_size,
                        const std::complex<float> *aterm,
                        const int *aterm_offsets,
                        const float *spheroidal);

                    virtual void degrid_visibilities(
                        std::complex<float> *visibilities,
                        const float *uvw,
                        const float *wavenumbers,
                        const int *baselines,
                        const std::complex<float> *grid,
                        const float w_offset,
                        const int kernel_size,
                        const std::complex<float> *aterm,
                        const int *aterm_offsets,
                        const float *spheroidal);

                    virtual void transform(DomainAtoDomainB direction,
                                           std::complex<float>* grid);

                private:
                    #if REDUCE_HOST_MEMORY
                    std::vector<cl::Buffer*> h_visibilities_;
                    std::vector<cl::Buffer*> h_uvw_;
                    #else
                    cl::Buffer h_visibilities;
                    cl::Buffer h_uvw;
                    #endif
                    std::vector<cl::Buffer*> h_grid_;

                    void init_benchmark();
                    bool enable_benchmark = false;
                    int nr_repetitions = 1;
            }; // class Generic

        } // namespace opencl
    } // namespace proxy
} // namespace idg
#endif
