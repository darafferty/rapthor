/**
 *  \class Generic
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CUDA_GENERIC_H_
#define IDG_CUDA_GENERIC_H_

#include <vector>

#include "idg-cuda.h"

using namespace idg::proxy::cuda;

namespace idg {
    namespace proxy {
        namespace cuda {
            class Generic : public Proxy {
                public:
                    /// Constructor
                    Generic(
                        Parameters params,
                        ProxyInfo info = default_info());

                    /// Destructor
                    ~Generic() = default;

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
                        const float *spheroidal) override;

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
                        const float *spheroidal) override;

                    virtual void transform(DomainAtoDomainB direction,
                                           std::complex<float>* grid) override;

                public:
                    void print_compiler_flags();
                    void print_devices();
                    std::vector<DeviceInstance*> get_devices();

                protected:
                    void init_devices();
                    static ProxyInfo default_info();

                protected:
                    ProxyInfo &info;
                    std::vector<DeviceInstance*> devices;

                public:
                    uint64_t sizeof_subgrids(int nr_subgrids);
                    uint64_t sizeof_uvw(int nr_baselines);
                    uint64_t sizeof_visibilities(int nr_baselines);
                    uint64_t sizeof_metadata(int nr_subgrids);
                    uint64_t sizeof_grid();
                    uint64_t sizeof_wavenumbers();
                    uint64_t sizeof_aterm();
                    uint64_t sizeof_spheroidal();

            }; // class Generic

        } // namespace cuda
    } // namespace proxy
} // namespace idg
#endif
