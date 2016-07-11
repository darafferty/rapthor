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

#include <cstdio> // remove()
#include <cstdlib>  // rand()
#include <ctime> // time() to init srand()
#include <complex>
#include <sstream>
#include <memory>
#include <exception>
#include <vector>

#include <dlfcn.h> // dlsym()
#include <omp.h> // omp_get_wtime
#include <libgen.h> // dirname() and basename()
#include <unistd.h> // rmdir()
#include <fftw3.h> // FFTW_BACKWARD, FFTW_FORWARD

#include "idg-config.h"
#include "idg-common.h"

#include "Kernels.h"


namespace idg {
    namespace proxy {
        namespace cpu {

            class CPU : public Proxy {
                public:
                    /// Constructors
                    CPU(Parameters params,
                        Compiler compiler,
                        Compilerflags flags,
                        ProxyInfo info);

                    // Disallow assignment and pass-by-value
                    CPU& operator=(const CPU& rhs) = delete;
                    CPU(const CPU& v) = delete;

                    /// Destructor
                    virtual ~CPU();

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

                    virtual void transform(
                        DomainAtoDomainB direction,
                        std::complex<float>* grid) override;

                // Low level routines
                public:
                    virtual void grid_onto_subgrids(
                        const Plan& plan,
                        const float w_offset,
                        const float *uvw,
                        const float *wavenumbers,
                        const std::complex<float> *visibilities,
                        const float *spheroidal,
                        const std::complex<float> *aterm,
                        std::complex<float> *subgrids);

                    virtual void add_subgrids_to_grid(
                        const Plan& plan,
                        const std::complex<float> *subgrids,
                        std::complex<float> *grid);

                    virtual void split_grid_into_subgrids(
                        const Plan& plan,
                        std::complex<float> *subgrids,
                        const std::complex<float> *grid);

                    virtual void degrid_from_subgrids(
                        const Plan& plan,
                        const float w_offset,
                        const float *uvw,
                        const float *wavenumbers,
                        std::complex<float> *visibilities,
                        const float *spheroidal,
                        const std::complex<float> *aterm,
                        const std::complex<float> *subgrids);

                    void fftshift(int nr_polarizations, std::complex<float> *grid);
                    void ifftshift(int nr_polarizations, std::complex<float> *grid);
                    void fftshift(std::complex<float> *array);
                    void ifftshift(std::complex<float> *array);

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

                    PowerSensor *powerSensor;
            }; // class CPU
        } // namespace cpu
    } // namespace proxy
} // namespace idg

#endif
