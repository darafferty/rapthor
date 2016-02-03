/**
 *  \class OpenCL
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_OPENCL_H_
#define IDG_OPENCL_H_

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <CL/cl.h>

#include "idg-common.h"
#include "Kernels.h"


namespace idg {
    namespace proxy {
        namespace opencl {
            /*
                Power measurement
            */
            static PowerSensor powerSensor;

            class OpenCL : public Proxy {
                public:
                    /// Constructors
                    OpenCL(Parameters params,
                        unsigned deviceNumber = 0,
                        Compilerflags flags = default_compiler_flags());

                    ~OpenCL();

                    // Get default values
                    static std::string default_compiler_flags();

                    // Get parameters of proxy
                    const Parameters& get_parameters() const { return mParams; }

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
                   uint64_t sizeof_subgrids(int nr_subgrids);
                   uint64_t sizeof_uvw(int nr_baselines);
                   uint64_t sizeof_visibilities(int nr_baselines);
                   uint64_t sizeof_metadata(int nr_subgrids);
                   uint64_t sizeof_grid();
                   uint64_t sizeof_wavenumbers();
                   uint64_t sizeof_aterm();
                   uint64_t sizeof_spheroidal();

                protected:
                    ProxyInfo default_proxyinfo(std::string srcdir, std::string tmpdir);

                    void compile(Compilerflags flags);
                    void parameter_sanity_check();

                    // data
                    cl::Context context;
                    cl::Device device;

                    // store cl::Program instances for every kernel
                    std::vector<cl::Program*> programs;
                    std::map<std::string,int> which_program;

            }; // class OpenCL
        } // namespace opencl
    } // namespace proxy
} // namespace idg

#endif
