/**
 *  \class CUDA
 *
 *  \brief Class for ...
 *
 *  Have a more detailed description here
 *  This will be included by a user, so detail usage...
 */

#ifndef IDG_CUDA_H_
#define IDG_CUDA_H_

#include <cstdio> // remove()
#include <cstdlib>  // rand()
#include <ctime> // time() to init srand()
#include <complex>
#include <sstream>
#include <memory>
#include <dlfcn.h> // dlsym()
#include <omp.h> // omp_get_wtime
#include <libgen.h> // dirname() and basename()
#include <unistd.h> // rmdir()
#include <vector>
#include <map>

#include <fftw3.h> // FFTW_BACKWARD, FFTW_FORWARD

#include <cuda.h>

#include "CU.h"

#include "idg-common.h"
#include "Kernels.h"
#include "DeviceInstance.h"

#define REUSE_HOST_MEMORY 0

namespace idg {
    namespace proxy {
        namespace cuda {
            /*
                Power measurement
            */
            static PowerSensor powerSensor;

            class PowerRecord {
                public:
                void enqueue(cu::Stream &stream);
                static void getPower(CUstream, CUresult, void *userData);
                PowerSensor::State state;
                cu::Event event;
            };

            class CUDA : public Proxy {
            public:
                /// Constructors
                CUDA(
                    Parameters params,
                    unsigned deviceNumber = 0,
                    ProxyInfo info = default_info()
                );

                ~CUDA();

                // Get default values
                static ProxyInfo default_info();
                static std::string default_compiler();
                static std::string default_compiler_flags();

                // Get parameters of proxy
                const Parameters& get_parameters() const { return mParams; }
                const ProxyInfo& get_info() const { return mInfo; }

                cu::Device& get_device() const {
                    return *device;
                }

                cu::Context& get_context() const {
                    return *context;
                }


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
                std::unique_ptr<kernel::cuda::Gridder> get_kernel_gridder() const;
                std::unique_ptr<kernel::cuda::Degridder> get_kernel_degridder() const;
                std::unique_ptr<kernel::cuda::GridFFT> get_kernel_fft() const;
                std::unique_ptr<kernel::cuda::Adder> get_kernel_adder() const;
                std::unique_ptr<kernel::cuda::Splitter> get_kernel_splitter() const;
                std::unique_ptr<kernel::cuda::Scaler> get_kernel_scaler() const;

            protected:
                virtual dim3 get_block_gridder() const = 0;
                virtual dim3 get_block_degridder() const = 0;
                virtual dim3 get_block_adder() const = 0;
                virtual dim3 get_block_splitter() const = 0;
                virtual dim3 get_block_scaler() const = 0;
                virtual int get_gridder_batch_size() const = 0;
                virtual int get_degridder_batch_size() const = 0;

            protected:
                static std::string make_tempdir();
                static ProxyInfo default_proxyinfo(std::string srcdir, std::string tmpdir);

                void compile(Compiler compiler, Compilerflags flags);
                void parameter_sanity_check();
                void load_shared_objects();
                void find_kernel_functions();
                virtual std::string append(Compilerflags flags) const = 0;

                void init_cuda(unsigned deviceNumber = 0);
                void compile_kernels(Compiler compiler, Compilerflags flags);
                void init_powersensor();

                // data
                cu::Device *device;
                cu::Context *context;
                ProxyInfo mInfo; // info about shared object files

                // store the ptr to Module, which each loads an .ptx-file
                std::vector<cu::Module*> modules;
                std::map<std::string,int> which_module;

            }; // class CUDA
        } // namespace cuda
    } // namespace proxy
} // namespace idg

#endif
