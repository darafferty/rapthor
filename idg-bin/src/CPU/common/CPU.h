/**
 *  \class CPU
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CPU_H_
#define IDG_CPU_H_

#include <dlfcn.h>
#include <memory>

#include <fftw3.h> // FFTW_BACKWARD, FFTW_FORWARD

#include "idg-common.h"
#include "Kernels.h"


namespace idg {
    namespace proxy {
        namespace cpu {
            // Power sensor
            static LikwidPowerSensor *powerSensor;

            class CPU : public Proxy {
                public:
                    /// Constructors
                    CPU(Parameters params,
                        Compiler compiler = default_compiler(),
                        Compilerflags flags = default_compiler_flags(),
                        ProxyInfo info = default_info());

                    /// Copy constructor
                    //CPU(const CPU& v) = delete;

                    /// Destructor
                    virtual ~CPU();

                    /// Assignment
                    CPU& operator=(const CPU& rhs) = delete;

                    // Get default values
                    static ProxyInfo default_info();
                    static std::string default_compiler();
                    static std::string default_compiler_flags();

                    // Get parameters of proxy
                    const Parameters& get_parameters() const { return mParams; }
                    const ProxyInfo& get_info() const { return mInfo; }

                    // High level interface, inherited from Proxy
                    virtual void grid_visibilities(
                        const std::complex<float> *visibilities,
                        const float *uvw,
                        const float *wavenumbers,
                        const int *baselines,
                        std::complex<float> *grid,
                        const float w_offset,
                        const std::complex<float> *aterm,
                        const float *spheroidal) override;

                    virtual void degrid_visibilities(
                        std::complex<float> *visibilities,
                        const float *uvw,
                        const float *wavenumbers,
                        const int *baselines,
                        const std::complex<float> *grid,
                        const float w_offset,
                        const std::complex<float> *aterm,
                        const float *spheroidal) override;

                    virtual void transform(DomainAtoDomainB direction,
                                           std::complex<float>* grid) override;

                // Low level routines
                public:
                    virtual void grid_onto_subgrids(
                        unsigned nr_subgrids,
                        float w_offset,
                        float *uvw,
                        float *wavenumbers,
                        std::complex<float> *visibilities,
                        float *spheroidal,
                        std::complex<float> *aterm,
                        int *metadata,
                        std::complex<float> *subgrids);

                    virtual void add_subgrids_to_grid(
                        unsigned nr_subgrids,
                        int *metadata,
                        std::complex<float> *subgrids,
                        std::complex<float> *grid);

                    virtual void split_grid_into_subgrids(
                        unsigned nr_subgrids,
                        int *metadata,
                        std::complex<float> *subgrids,
                        std::complex<float> *grid);

                    virtual void degrid_from_subgrids(
                        unsigned nr_subgrids,
                        float w_offset,
                        float *uvw,
                        float *wavenumbers,
                        std::complex<float> *visibilities,
                        float *spheroidal,
                        std::complex<float> *aterm,
                        int *metadata,
                        std::complex<float> *subgrids);


                    // Auxiliary: additional set and get methods
                    // Note: the abstract proxy provides less,
                    // as it does not know that how the high level
                    // routines are split up in subroutines
                public:
                    void set_job_size_gridder(unsigned int js) {
                        mParams.set_job_size_gridder(js); }
                    void set_job_size_adder(unsigned int js) {
                        mParams.set_job_size_adder(js); }
                    void set_job_size_splitter(unsigned int js) {
                        mParams.set_job_size_splitter(js); }
                    void set_job_size_degridder(unsigned int js) {
                        mParams.set_job_size_degridder(js);
                    }

                public:
                    virtual std::unique_ptr<idg::kernel::cpu::Gridder> get_kernel_gridder() const;
                    virtual std::unique_ptr<idg::kernel::cpu::Degridder> get_kernel_degridder() const;
                    virtual std::unique_ptr<idg::kernel::cpu::Adder> get_kernel_adder() const;
                    virtual std::unique_ptr<idg::kernel::cpu::Splitter> get_kernel_splitter() const;
                    virtual std::unique_ptr<idg::kernel::cpu::GridFFT> get_kernel_fft() const;

                    LikwidPowerSensor::State read_power() {
                        return cpu::powerSensor->read();
                    }


                protected:
                    static std::string make_tempdir();
                    static ProxyInfo default_proxyinfo(std::string srcdir,
                                                       std::string tmpdir);

                    void compile(Compiler compiler, Compilerflags flags);
                    void parameter_sanity_check();
                    void load_shared_objects();
                    void find_kernel_functions();

                    // data
                    ProxyInfo mInfo; // info about shared object files

                    // store the ptr to Module, which each loads an .so-file
                    std::vector<runtime::Module*> modules;
                    std::map<std::string,int> which_module;
            }; // class CPU
        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
