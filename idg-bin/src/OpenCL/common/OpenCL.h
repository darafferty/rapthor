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

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <memory>

#include "idg-common.h"
#include "idg-powersensor.h"
#include "Kernels.h"
#include "Util.h"


namespace idg {
    namespace proxy {
        namespace opencl {
            class OpenCL : public Proxy {
                public:
                    /// Constructors
                    OpenCL(
                        Parameters params,
                        unsigned deviceNumber = 0);

                    ~OpenCL();

                    // Get default values
                    std::string default_compiler_flags();

                    // Get parameters of proxy
                    const Parameters& get_parameters() const { return mParams; }

                    // Get context of proxy
                    cl::Context get_context() const { return context; }

                    // Get device of proxy
                    cl::Device get_device() const { return device; }

                    // Get power sensor of proxy
                    PowerSensor *get_powersensor() const { return powerSensor; }

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

                public:
                   virtual std::unique_ptr<kernel::opencl::Gridder> get_kernel_gridder() const = 0;
                   virtual std::unique_ptr<kernel::opencl::Degridder> get_kernel_degridder() const = 0;
                   virtual std::unique_ptr<kernel::opencl::GridFFT> get_kernel_fft() const = 0;
                   virtual std::unique_ptr<kernel::opencl::Scaler> get_kernel_scaler() const = 0;
                   virtual std::unique_ptr<kernel::opencl::Adder> get_kernel_adder() const = 0;
                   virtual std::unique_ptr<kernel::opencl::Splitter> get_kernel_splitter() const = 0;

                public:
                    PowerSensor *powerSensor;

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
