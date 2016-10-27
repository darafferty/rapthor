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

#include "../common/OpenCL.h"

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
                    PowerSensor *hostPowerSensor;

                    void init_benchmark();
                    bool enable_benchmark = false;
                    int nr_repetitions = 1;
            }; // class Generic

        } // namespace opencl
    } // namespace proxy
} // namespace idg
#endif
